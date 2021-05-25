AutoRL Starting Kit
======================================

## Contents
`ingestion/`: The code and libraries used on Codalab to run your submmission.

`scoring/`: The code and libraries used on Codalab to score your submmission.

`code_submission/`: An example of code submission you can use as template.

`data/`: We provide 3 public environments for participants in `data/public`, 
each with 5 train configuration files and 5 test configuration files. You can use these environments to
test and develop your own solutions.

`baseline/`: Two baselines we provided.

`run_local_test.py`: A python script to simulate the runtime in codalab

## Local development and testing
1. To make your own submission to AutoRL challenge, you need to modify the
   file `solution.py` in `code_submission/`, which implements your algorithm.
   You need to define `Trainer` class in `solution.py`. The design
   of interface is as follows

```python
class Trainer:
    def __init__(self, Env, conf_list):
        # Env, the environment class
        #conf_list, array of configuration file paths (str)
        pass
    
    # your training solution code
    def train(self, time):     
        # time: time budget, in sec
        # return an instance of Agent
        return agent # see the website page Submission for more information on agent
```

The output of ``Trainer.train()`` is an agent, which is an object of the ``Agent`` class.

The ``Agent`` class should have an ``act()`` method:

```python
class Agent:
    def act(self, machine_status, job_status, time, job_list):  
        ...
        return job_assignment	# outputs job assignment
```

For more details about the interface, please see the competition website.

2. Test the algorithm on your local computer using Docker,
   in the exact same environment as on the CodaLab challenge platform. Advanced
   users can also run local test without Docker, if they install all the required
   packages.
3. If you are new to docker, install docker from https://docs.docker.com/get-started/.
   Then, at the shell, run:

```shell
cd path/to/autorl_starting_kit/
(CPU)docker run -it --rm -v "$(pwd):/app/program" -w /app/program dodow/autorl:v2
(GPU)docker run --gpus=2 -it --rm -v "$(pwd):/app/program" -w /app/program dodow/autorl:v2
```

Please note that for running docker with GPU support, you need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) first.

The option `-v "$(pwd):/app/program"` mounts current directory
(`autorl_starting_kit/`) as `/app/program`. If you want to mount other
directories on your disk, please replace `$(pwd)` with your own directory.

The Docker image
```
dodow/autorl:v2
```

4. You will then be able to run the `ingestion program` (to train and produce test rewards)
   and the `scoring program` (to evaluate your predictions) on toy sample data.
   In the AutoRL challenge, both two programs will run in parallel to give
   feedback. So we provide a Python script to simulate this behavior. To test locally, run:

```shell
python run_local_test.py --dataset_dir=./data/public/01
```

If the program exits without any errors, you can find the final score from the terminal's stdout of your solution.
Also you can view the score by opening the `scoring_output/scores.txt`.

The full usage is

```shell
python run_local_test.py --dataset_dir=./data/public/01 --code_dir=./code_submission
```

You can change the argument `dataset_dir` to other datasets (e.g. the three
practice environment configurations we provide). On the other hand, you can also modify the directory containing your other sample code.

## Baselines
We provide 2 baselines:
1. Reinforcement Learning with PPO;
   Attention: RL baseline uses 6 CPU and 1 GPU, you can modify "`num_workers`" and "`num_gpus`" with your local resource in line 26 and line 27 in `./baseline/rl/solution.py`.
2. Advanced pending-time-first rule-based policy.

To try out baselines, you need to copy the baseline code into `code_submission/`.

For the priority-rule-based baseline, do
```shell
# run Advanced pending-time-first baseline
cp ./basline/adv_wwsqt/*.py ./code_submission
```

For the reinforcement learning baseline, do
```shell
# run Reinforcement learning baseline
cp ./basline/rl/*.py ./code_submission
export PYTHONPATH=$PYTHONPATH:/app/program/code_submission
# NOTE: During online evaluation, code_submission will be appended to $PYTHONPATH automatically
```

*NOTE: the reinforcement learning baseline has only been tested on a workstation with
8 CPU cores, 1 GPU, and 32 GB Memory.*

## Prepare a ZIP file for submission on CodaLab
Zip the contents of `code_submission`(or any folder containing
your `solution.py` file) without the directory structure:

```shell
cd code_submission/
zip -r mysubmission.zip *
```

then use the "Upload a Submission" button to make a submission to the
competition page on challenge platform.

Tip: to look at what's in your submission zip file without unzipping it, you
can do

```shell
unzip -l mysubmission.zip
```

## Using third party packages

You can include a `requirements.txt` file in `code_submission/` to install third party packages.
The platform will automatically run `pip install -r requirements.txt` before evaluation.

## Report bugs and create issues

If you run into bugs or issues when using this starting kit, please contact us via:
<autorl2021@4paradigm.com>
