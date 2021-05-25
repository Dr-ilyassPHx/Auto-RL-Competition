# pylint: disable=logging-fstring-interpolation
"""scoring function for autograph"""

import argparse
import datetime
import glob
import math
import os
from os.path import join, isfile
import logging
import sys
import time
import yaml
import pickle

import psutil
import pandas as pd

import mmap
import contextlib
from filelock import FileLock


# Verbosity level of logging.
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
#  VERBOSITY_LEVEL = 'INFO'
VERBOSITY_LEVEL = 'INFO'
WAIT_TIME = 30
MAX_TIME_DIFF = datetime.timedelta(seconds=600)
DEFAULT_SCORE = -99999999
# SOLUTION_FILE = 'test_label.tsv'
N_TEST = 10
TMAX = 3500



def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
        2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


LOGGER = get_logger(VERBOSITY_LEVEL)


def _here(*args):
    """Helper function for getting the current directory of the script."""
    here_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(join(here_dir, *args))


def _get_solution(solution_dir):
    """Get the solution array from solution directory."""
    file_list = os.listdir(solution_dir)

    solution_file = [join(solution_dir, file) for file in file_list]
    return solution_file


def _get_rewards(prediction_dir):
    with open(join(prediction_dir,'rewards.pkl'), 'rb') as f:
        rewards = pickle.load(f)
    return rewards


def _get_score(prediction_dir):
    """get score"""

    LOGGER.info('===== read rewards')
    rewards = _get_rewards(prediction_dir)
    return sum(rewards)/len(rewards)


def _update_score(args, duration):
    score = _get_score(prediction_dir=args.prediction_dir)
    # Update learning curve page (detailed_results.html)
    _write_scores_html(args.score_dir)
    # Write score
    LOGGER.info('===== write score')
    write_score(args.score_dir, score, duration)
    LOGGER.info(f"reward: {score}")
    return score

def _update_error_score(args, duration):
    score = -99999999
    # Update learning curve page (detailed_results.html)
    _write_scores_html(args.score_dir)
    # Write score
    LOGGER.info('===== write score')
    write_score(args.score_dir, score, duration)
    LOGGER.info(f"reward: {score}")
    return score


def _init_scores_html(detailed_results_filepath):
    html_head = ('<html><head> <meta http-equiv="refresh" content="5"> '
                 '</head><body><pre>')
    html_end = '</pre></body></html>'
    with open(detailed_results_filepath, 'a') as html_file:
        html_file.write(html_head)
        html_file.write("Starting training process... <br> Please be patient. "
                        "Learning curves will be generated when first "
                        "predictions are made.")
        html_file.write(html_end)


def _write_scores_html(score_dir, auto_refresh=True, append=False):
    filename = 'detailed_results.html'
    if auto_refresh:
        html_head = ('<html><head> <meta http-equiv="refresh" content="5"> '
                     '</head><body><pre>')
    else:
        html_head = """<html><body><pre>"""
    html_end = '</pre></body></html>'
    if append:
        mode = 'a'
    else:
        mode = 'w'
    filepath = join(score_dir, filename)
    with open(filepath, mode) as html_file:
        html_file.write(html_head)
        html_file.write(html_end)
    LOGGER.debug(f"Wrote learning curve page to {filepath}")


def write_score(score_dir, score, duration):
    """Write score and duration to score_dir/scores.txt"""
    score_filename = join(score_dir, 'scores.txt')
    with open(score_filename, 'w') as ftmp:
        ftmp.write(f'score: {score}\n')
        ftmp.write(f'Duration: {duration}\n')
    LOGGER.debug(f"Wrote to score_filename={score_filename} with "
                 f"score={score}, duration={duration}")


class IngestionError(Exception):
    """Ingestion error"""


class ScoringError(Exception):
    """scoring error"""

class IngestionMonitor():

    def __init__(self, ingestion_output_dir):
        self._startfile_path = join(ingestion_output_dir, 'start.txt')
        self._endfile_path = join(ingestion_output_dir, 'end.yaml')

    # Wait 30 seconds for ingestion to start and write 'start.txt',
    # and establish shared memory mappings to start.txt.
    # Otherwise, raise an exception.
    def setup(self):
        LOGGER.info('===== wait for ingestion to start')
        for _ in range(WAIT_TIME):
            if self.detect_startfile():
                LOGGER.info('===== detect alive ingestion')
                break
            time.sleep(1)
        else:
            raise IngestionError("[-] Failed: scoring didn't detected the start "
                                 f"of ingestion after {WAIT_TIME} seconds.")

        startfile = open(self._startfile_path, 'r')
        self._startfile = startfile
        self._shm = mmap.mmap(startfile.fileno(), 0, access=mmap.ACCESS_READ)

    def wait_ingestion_exit(self):
        while True:
            try:
                self._shm.seek(0)
                last_time = self._shm.readline()
                last_time = float(last_time)
                last_time = datetime.datetime.fromtimestamp(last_time)
                LOGGER.debug(f"***** last time from ingestion = {last_time}")

                current_time = datetime.datetime.now()
                timediff = current_time - last_time
                if timediff > MAX_TIME_DIFF:
                    LOGGER.info(f"***** timediff exceed MAX_TIME_DIFF, timediff = {timediff}")
                    break
            except Exception:
                LOGGER.info(f"the content of start.txt is: {self._startfile.read()}")
                self._shm.close()
                self._startfile.close()
                raise

            if self.detect_endfile():
                LOGGER.info('detect end.yaml')
                break

            time.sleep(1)
        else:
            self._shm.close()
            self._startfile.close()

    def detect_startfile(self):
        return isfile(self._startfile_path)

    def detect_endfile(self):
        return isfile(self._endfile_path)

    def get_ingestion_info(self):
        lockfile = self._endfile_path + '.lock'
        with FileLock(lockfile):
            with open(self._endfile_path, 'r') as ftmp:
                ingestion_info = yaml.safe_load(ftmp)
        return ingestion_info


def get_ingestion_info(prediction_dir):
    """get ingestion information"""
    ingestion_info = None
    endfile_path = os.path.join(prediction_dir, 'end.yaml')

    if not os.path.isfile(endfile_path):
        raise IngestionError("[-] No end.yaml exist, ingestion failed")

    LOGGER.info('===== Detected end.yaml file, get ingestion information')
    with open(endfile_path, 'r') as ftmp:
        ingestion_info = yaml.safe_load(ftmp)

    return ingestion_info


def get_ingestion_pid(prediction_dir):
    """get ingestion pid"""
    # Wait 60 seconds for ingestion to start and write 'start.txt',
    # Otherwise, raise an exception.
    wait_time = 60
    startfile = os.path.join(prediction_dir, 'start.txt')
    lockfile = os.path.join(prediction_dir, 'start.txt.lock')

    for i in range(wait_time):
        if os.path.exists(startfile):
            with FileLock(lockfile):
                with open(startfile, 'r') as ftmp:
                    ingestion_pid = ftmp.read()
                    LOGGER.info(
                        f'Detected the start of ingestion after {i} seconds.')
                    return int(ingestion_pid)
        else:
            time.sleep(1)
    raise IngestionError(f'[-] Failed: scoring didn\'t detected the start of'
                         'ingestion after {wait_time} seconds.')


def is_process_alive(ingestion_pid):
    """detect ingestion alive"""
    try:
        os.kill(ingestion_pid, 0)
    except OSError:
        return False
    else:
        return True


def _parse_args():
    # Default I/O directories:
    root_dir = _here(os.pardir)
    default_solution_dir = join(root_dir, "sample_data")
    default_prediction_dir = join(root_dir, "sample_result_submission")
    default_score_dir = join(root_dir, "scoring_output")
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution_dir', type=str,
                        default=default_solution_dir,
                        help=("Directory storing the solution with true "
                              "labels, e.g. adult.solution."))
    parser.add_argument('--prediction_dir', type=str,
                        default=default_prediction_dir,
                        help=("Directory storing the predictions. It should"
                              "contain e.g. [start.txt, adult.predict_0, "
                              "adult.predict_1, ..., end.yaml]."))
    parser.add_argument('--score_dir', type=str,
                        default=default_score_dir,
                        help=("Directory storing the scoring output e.g. "
                              "`scores.txt` and `detailed_results.html`."))
    args = parser.parse_args()
    LOGGER.debug(f"Parsed args are: {args}")
    LOGGER.debug("-" * 50)
    LOGGER.debug(f"Using solution_dir: {args.solution_dir}")
    LOGGER.debug(f"Using prediction_dir: {args.prediction_dir}")
    LOGGER.debug(f"Using score_dir: {args.score_dir}")
    return args


def _init(args):
    if not os.path.isdir(args.score_dir):
        os.mkdir(args.score_dir)
    detailed_results_filepath = join(
        args.score_dir, 'detailed_results.html')
    # Initialize detailed_results.html
    _init_scores_html(detailed_results_filepath)


def _finalize(score, scoring_start):
    """finalize the scoring"""
    # Use 'end.yaml' file to detect if ingestion program ends
    duration = time.time() - scoring_start
    LOGGER.info(
        "[+] Successfully finished scoring! "
        f"Scoring duration: {duration:.2} sec. "
        f"The score of your algorithm on the task is: {score:.6}.")

    LOGGER.info("[Scoring terminated]")


def main():
    """main entry"""
    scoring_start = time.time()
    LOGGER.info('===== init scoring program')
    args = _parse_args()
    _init(args)
    score = DEFAULT_SCORE
    ingestion_monitor = IngestionMonitor(args.prediction_dir)
    ingestion_monitor.setup()
    # Moniter training processes, stop when ingestion stop or detect endfile
    LOGGER.info('===== wait for the exit of ingestion or end.yaml file')
    ingestion_monitor.wait_ingestion_exit()

    if not ingestion_monitor.detect_endfile():
        LOGGER.error("no end.yaml exist, ingestion failed")
        score = _update_error_score(args, 0)
        raise RuntimeError
    else:
        LOGGER.info('===== end.yaml file detected, get ingestion information')
        # Compute/write score
        ingestion_info = ingestion_monitor.get_ingestion_info()
        duration = ingestion_info['ingestion_duration']
        score = _update_score(args, duration)

    _finalize(score, scoring_start)


if __name__ == "__main__":
    main()
