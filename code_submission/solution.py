import time
import random
from collections import defaultdict

class Trainer:
    def __init__(self, Env, conf_list):
        self.conf_list = conf_list
        self.Env = Env
        self.checkpoint = None
        self.iter = 0

    def train(self, run_time):
        env = self.Env(self.conf_list[0])
        obs = env.reset()
        return Agent(env.job_types)

class Agent:
    def __init__(self, job_types):
        self.job_types = job_types

    
    def act(self, machine_status, job_status, time, job_list):
        action = {}
        for machine in job_list:
            job = self.adv_wwsqt(machine, machine_status, job_status, time, job_list[machine])
            if job is not None:
                for mm in job_list:
                    try:
                        job_list[mm].remove(a)
                    except:
                        pass
                    finally:
                        pass
                action[machine] = job
        return action

    def adv_wwsqt(self, machine, machine_status, job_status, time, job_list):

        def get_next_op_info(job):
            if job_status[job]['status'] == 'to_arrive':
                job_type = job_status[job]['type']
                next_op = self.job_types[job_type][0]
                return {'machine':'A', 'next_max_pending_time':next_op['max_pend_time']}
            else:
                job_type = job_status[job]['type']
                now_op = job_status[job]['op']
                for op_idx, op in enumerate(self.job_types[job_type]):
                    if op['op_name'] == now_op:
                        break
                next_op = self.job_types[job_type][op_idx+1] if op_idx < len(self.job_types[job_type]) - 1 else None
                next_op_info = {'machine':next_op['machine_type'], 'next_max_pending_time':next_op['max_pend_time']} if next_op is not None \
                    else {'machine':None, 'next_max_pending_time':None}
                return next_op_info

        if len(job_list) == 0:
            return None
        else:
            sorted_list = [a for a in job_list if job_status[a]['priority']>0]
            if len(sorted_list) == 0:
                machine_type = machine_status[machine]['type']
                require_time = {}
                for job in job_status:
                    next_op_info = get_next_op_info(job)
                    if (job_status[job]['status'] == 'work' or job_status[job]['status'] == 'to_arrive') and next_op_info['machine'] == machine_type and job_status[job]['priority'] > 0:
                        require_time[job] = job_status[job]["remain_process_time"] + next_op_info['next_max_pending_time']
                if len(require_time) == 0:
                    return sorted(job_list, key=lambda x: (job_status[x]['remain_process_time'], job_status[x]['remain_pending_time']))[0]
                else:
                    return None
            else:
                return sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/job_status[x]['priority'],job_status[x]['remain_process_time']))[0]

        









