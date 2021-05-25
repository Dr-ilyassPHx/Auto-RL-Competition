import numpy as np
from collections import defaultdict
from copy import deepcopy


class Agent:
    def __init__(self, trainer, env):
        self.trainer = trainer
        self.env = env()
        self.env.reset()
        self.last_valid_action = None

    def act(self, machine_status, job_status, time, job_list):
        self.machine_status = machine_status
        self.job_status = job_status
        self.time = time
        self.job_list = job_list

        self.valid_action = {}
        self.valid_machine_type = defaultdict(list)
        for machine in self.machine_status:
            if len(self.machine_status[machine]['job_list']) != 0:
                self.valid_action[machine] = self.machine_status[machine]['job_list']

        self.job_loc = {}

        machine_dict = {
            'A':0,
            'B':1,
            'C':2,
            'D':3
        }
        for job in self.job_status:
            job_op = self.job_status[job]['op']
            job_type = self.job_status[job]['type']

            for op in self.env.env.job_types[job_type]:
                if op['op_name'] == job_op:
                    op_machine = op['machine_type']
                    self.job_loc[job] = machine_dict[op_machine]
                    break

        for machine in self.valid_action:
            self.valid_machine_type[self.machine_status[machine]['type']].append(machine)



        if self._valid_action_step():
            self.last_valid_action = deepcopy(self.valid_action)

            return {}

        else:
            self.last_valid_action = deepcopy(self.valid_action)
            self.get_candidates()
            obs = {}
            for machine_type in self.valid_machine_type:
                if len(self.valid_machine_type[machine_type]) > 0:
                    agent_id = self.valid_machine_type[machine_type][0]
                    obs[agent_id] = {
                        'obs':self.gen_observation(agent_id),
                        'action_mask':self.get_action_mask(agent_id)
                    }

            rl_actions = {}
            for agent_id in obs:
                rl_actions[agent_id] = self.trainer.compute_action(obs[agent_id], explore=False)

            step_actions = {}
            eval_actions = {}
            for key in rl_actions:
                if rl_actions[key] == 0:
                    continue
                elif rl_actions[key] == 1:
                    a_ = self.qtfirst(key)
                    if a_ is not None:
                        eval_actions[key] = a_
                        self.job_list[key] = []
                        del self.valid_action[key]
                        for tool in self.valid_action:
                            if a_ in self.valid_action[tool]:
                                self.valid_action[tool].remove(a_)
                elif rl_actions[key] == 2:
                    a_ = self.ptfirst(key)
                    if a_ is not None:
                        eval_actions[key] = a_
                        self.job_list[key] = []
                        del self.valid_action[key]
                        for tool in self.valid_action:
                            if a_ in self.valid_action[tool]:
                                self.valid_action[tool].remove(a_)
            for k in eval_actions:
                self.job_loc[eval_actions[k]] += 1
            step_actions.update(eval_actions)

            while self.check_real_step():
                for key in eval_actions:
                    job = eval_actions[key]
                    for machine in self.machine_status:
                        if key == machine:
                            self.machine_status[machine].update({
                                'job_list':[],
                                'status':'work',
                                'job': job,
                            })
                        else:
                            if job in self.machine_status[machine]['job_list']:
                                self.machine_status[machine]['job_list'].remove(job)
                    self.job_status[job].update({
                        'status':'work',
                        'op': self.get_job_op(job, key),
                        'machine':key
                    })
                    self.job_list[key] = []

                for key in self.job_list:
                    if job in self.job_list[key]:
                        self.job_list[key].remove(job)


                self.valid_action = {}
                for machine in self.machine_status:
                    if machine in self.job_list and len(self.job_list[machine]) != 0:
                        self.valid_action[machine] = list(self.job_list[machine])

                self.get_candidates()

                obs = {}
                for machine_type in self.valid_machine_type:
                    if len(self.valid_machine_type[machine_type]) > 0:
                        agent_id = self.valid_machine_type[machine_type][0]
                        obs[agent_id] = {
                            'obs':self.gen_observation(agent_id),
                            'action_mask':self.get_action_mask(agent_id)
                        }

                rl_actions = {}
                for agent_id in obs:
                    rl_actions[agent_id] = self.trainer.compute_action(obs[agent_id], explore=False)

                eval_actions = {}
                for key in rl_actions:
                    if rl_actions[key] == 0:
                        continue
                    elif rl_actions[key] == 1:
                        a_ = self.qtfirst(key)
                        if a_ is not None:
                            eval_actions[key] = a_
                            self.job_list[key] = []
                            del self.valid_action[key]
                            for tool in self.valid_action:
                                if a_ in self.valid_action[tool]:
                                    self.valid_action[tool].remove(a_)
                    elif rl_actions[key] == 2:
                        a_ = self.ptfirst(key)
                        if a_ is not None:
                            eval_actions[key] = a_
                            self.job_list[key] = []
                            del self.valid_action[key]
                            for tool in self.valid_action:
                                if a_ in self.valid_action[tool]:
                                    self.valid_action[tool].remove(a_)
                for k in eval_actions:
                    self.job_loc[eval_actions[k]] += 1
                step_actions.update(eval_actions)

        return step_actions



    def qtfirst(self, agent_id):
        candi_list = []
        for job in self.valid_action[agent_id]:
            candi_list.append(job)
        if len(candi_list) == 0:
            return None
        else:
            sorted_list = [a for a in candi_list if self.job_status[a]['priority']>0]
            if len(sorted_list) > 0:
                a = sorted(sorted_list, key=lambda x: (self.job_status[x]['remain_pending_time']/self.job_status[x]['priority'],self.job_status[x]['remain_process_time']))[0]
            else:
                a = sorted(candi_list, key=lambda x: self.job_status[x]['remain_process_time'])[0]
            return a

    def ptfirst(self, agent_id):
        candi_list = []
        for job in self.valid_action[agent_id]:
            candi_list.append(job)
        if len(candi_list) == 0:
            return None
        else:
            a = sorted(candi_list, key=lambda x: (self.job_status[x]['remain_process_time'], self.job_status[x]['remain_pending_time']))[0]
            return a

    def get_candidates(self):
        self.candidates = {}
        for i in self.valid_action:
            self.candidates[i] = self._get_candidates(agent_id=i)

        self.lots_all = {}

        for i in range(len(self.env.rl_agent_id)):
            tool_id = self.env.rl_agent_id[i]
            if self.machine_status[tool_id]['type'] not in self.lots_all:
                self.lots_all[self.machine_status[tool_id]['type']] = []
            if tool_id in self.candidates:
                self.lots_all[self.machine_status[tool_id]['type']] += [list(self.candidates[tool_id])]
            else:
                self.lots_all[self.machine_status[tool_id]['type']] += [[]]*(self.env.action_size-1)



    def _get_candidates(self, agent_id):
        res = []
        for i in range(1,self.env.action_size):
            res.append(self.act_rules(agent_id, i))
        return res

    def act_rules(self, agent_id, action, filter_lots=[]):
        if agent_id not in self.valid_action:
            return []

        lots = [i for i in self.valid_action[agent_id] if i not in filter_lots]
        if len(lots) == 0:
            return []


        if action == 1:
            lot = self.qtfirst(agent_id)
        elif action == 2:
            lot = self.ptfirst(agent_id)

        return lot

    def get_attr(self, lots, attr):
        res = []
        if isinstance(lots, list) or isinstance(lots, set):
            for lot in lots:
                if attr == 'remain_pending_time':
                    if self.job_status[lot]['priority'] == 0:
                        res.append(99)
                    else:
                        res.append(self.job_status[lot][attr] / self.job_status[lot]['priority'])
                else:
                    res.append(self.job_status[lot][attr])
        else:
            lot = lots
            if attr == 'remain_pending_time':
                if self.job_status[lot]['priority'] == 0:
                    res.append(99)
                else:
                    res.append(self.job_status[lot][attr] / self.job_status[lot]['priority'])
            else:
                res.append(self.job_status[lot][attr])
        return res

    def gen_lots_features(self, lots):
        # length: 8
        ft = []
        pt = np.array(self.get_attr(lots, 'remain_process_time'))
        rqt = np.array(self.get_attr(lots, 'remain_pending_time'))
        ft.append(len(pt))

        if len(rqt) > 0 :
            ft.append(np.sum(rqt<=0))
        else:
            ft.append(0)

        if len(rqt) > 0:
            ft.append(np.mean(rqt))
            ft.append(np.max(rqt))
            ft.append(np.min(rqt))
        else:
            ft += [0,0,0]

        if len(pt) > 0:
            ft.append(np.mean(pt))
            ft.append(np.max(pt))
            ft.append(np.min(pt))
        else:
            ft += [0,0,0]

        return ft

    def check_real_step(self):
        for machine_type in self.valid_machine_type:
            if len(self.valid_machine_type[machine_type]) > 0:
                return False
        return True

    def get_job_op(self, job, machine):
        return self.job_status[job]['op']

    def get_action_mask(self, agent_id):
        res = np.ones(self.env.action_size)
        machine_type = self.machine_status[agent_id]['type']
        machine_dict = {
            'A':0,
            'B':1,
            'C':2,
            'D':3
        }
        can_wait = False

        for job in self.job_loc:
            if not job in self.valid_action[agent_id]:
                if self.job_loc[job] <= machine_dict[machine_type]:
                    can_wait = True

        res[0] = 1 if can_wait else 0

        return res


    def _valid_action_step(self):
        if self.last_valid_action is None:
            return False
        if len(self.valid_action) == 0:
            return True
        if len(self.last_valid_action) != len(self.valid_action):
            return False
        for key in self.last_valid_action:
            if key in self.valid_action:
                if set(self.valid_action[key]) != set(self.last_valid_action[key]):
                    return False
            else:
                return False
        return True

    def gen_observation(self, agent_id):
        '''
        :param agent_id:
        :return:
            features
            (length 92)
        '''
        agent_obs = []

        valid_actions = self.valid_action[agent_id] if agent_id in self.valid_action else []
        ft = self.gen_lots_features(valid_actions)
        agent_obs += ft

        ft = self.get_act_rule_feature(agent_id)
        agent_obs += ft

        ft = self.get_working_jobs_feature(agent_id)
        agent_obs += ft

        ft = self.get_pending_jobs_feature(agent_id)
        agent_obs += ft

        ft = self.get_machine_feature(agent_id)
        agent_obs += ft

        machine_type = self.machine_status[agent_id]['type']
        ft_dict = {
            'A':0,
            'B':1,
            'C':2,
            'D':3
        }
        ft = [0,0,0,0]
        ft[ft_dict[machine_type]] = 1
        agent_obs += ft
        return np.array(agent_obs)

    def get_act_rule_feature(self, agent_id):
        # length: 20
        ft = []
        ft += self.get_act_rule_job_feature(agent_id)
        ft += self.get_act_rule_machine_feature(agent_id)
        ft += self.get_act_rule_prev_feature(agent_id)

        return ft


    def get_act_rule_job_feature(self, agent_id):
        # length: 4
        ft = []
        if agent_id not in self.candidates:
            return [0,0,0,0]
        candi_jobs = self.candidates[agent_id]
        for job in candi_jobs:
            ft.append(self.job_status[job]['remain_process_time'])
            if self.job_status[job]['priority'] == 0:
                pending_time = 99
            else:
                pending_time = self.job_status[job]['remain_pending_time'] / self.job_status[job]['priority']
            ft.append(pending_time)
        return ft

    def get_act_rule_machine_feature(self, agent_id):
        # length: 8
        ft = []
        if agent_id not in self.candidates:
            return [0,0,0,0,0,0,0,0]
        candi_jobs = self.candidates[agent_id]
        for job in candi_jobs:
            machine_info = []
            for other_job in self.valid_action[agent_id]:
                if job != other_job and self.job_status[other_job]['remain_pending_time']-self.job_status[job]['remain_process_time']<0:
                    machine_info.append(self.job_status[other_job]['remain_pending_time']-self.job_status[job]['remain_process_time'])
            ft += [len(machine_info), max(machine_info), min(machine_info), sum(machine_info)/len(machine_info)] if len(machine_info) > 0 \
                else[0,0,0,0]

        return ft

    def get_act_rule_prev_feature(self, agent_id):
        # 动作规则下即将到此类机器的jobs（还在上一类机器中work或者还未arrive）中可能出现的qtime超时信息
        # length: 8
        def get_next_op_info(job):
            if self.job_status[job]['status'] == 'to_arrive':
                job_type = self.job_status[job]['type']
                next_op = self.env.env.job_types[job_type][0]
                return {'machine':'A', 'next_max_pending_time':next_op['max_pend_time']}
            else:
                job_type = self.job_status[job]['type']
                now_op = self.job_status[job]['op']
                for op_idx, op in enumerate(self.env.env.job_types[job_type]):
                    if op['op_name'] == now_op:
                        break
                next_op = self.env.env.job_types[job_type][op_idx+1] if op_idx < len(self.env.env.job_types[job_type]) - 1 else None
                next_op_info = {'machine':next_op['machine_type'], 'next_max_pending_time':next_op['max_pend_time']} if next_op is not None \
                    else {'machine':None, 'next_max_pending_time':None}
                return next_op_info


        working_job_dict = defaultdict(dict)
        for job in self.job_status:
            status = self.job_status[job]['status']
            if status == 'work' or status == 'to_arrive':
                next_op_info = get_next_op_info(job)
                if next_op_info['machine'] is not None:
                    working_job_dict[next_op_info['machine']][job] = {
                        'remain_process_time': self.job_status[job]['arrival'] if status == 'to_arrive' else self.job_status[job]['remain_process_time'],
                        'max_pending_time': next_op_info['next_max_pending_time']
                    }

        machine_type = self.machine_status[agent_id]['type']

        if agent_id not in self.candidates:
            return [0,0,0,0,0,0,0,0]

        qtime_first_info = []
        qtime_first_job = self.candidates[agent_id][0]

        qtime_process_time = self.job_status[qtime_first_job]['remain_process_time']

        for job in working_job_dict[machine_type]:
            next_op_job_info = working_job_dict[machine_type][job]
            if  next_op_job_info['remain_process_time'] + next_op_job_info['max_pending_time'] < qtime_process_time:
                qtime_first_info.append(qtime_process_time-(next_op_job_info['remain_process_time'] + next_op_job_info['max_pending_time']))


        ptime_first_info = []
        ptime_first_job = self.candidates[agent_id][1]
        ptime_process_time = self.job_status[ptime_first_job]['remain_process_time']

        for job in working_job_dict[machine_type]:
            next_op_job_info = working_job_dict[machine_type][job]
            if  next_op_job_info['remain_process_time'] + next_op_job_info['max_pending_time'] < ptime_process_time:
                ptime_first_info.append(ptime_process_time-(next_op_job_info['remain_process_time'] + next_op_job_info['max_pending_time']))

        ptime_info = [len(ptime_first_info), max(ptime_first_info), min(ptime_first_info), sum(ptime_first_info)/len(ptime_first_info)] if len(ptime_first_info) > 0 \
            else [0,0,0,0]
        qtime_info = [len(qtime_first_info), max(qtime_first_info), min(qtime_first_info), sum(qtime_first_info)/len(qtime_first_info)] if len(qtime_first_info) > 0 \
            else [0,0,0,0]

        return ptime_info + qtime_info

    def get_working_jobs_feature(self, agent_id):
        # length: 20
        ft = []
        pt_dict = {'A':[],'B':[], 'C':[], 'D':[]}
        for machine in self.machine_status:
            status = self.machine_status[machine]['status']
            if status == 'work':
                job = self.machine_status[machine]['job']
                pt_dict[self.machine_status[machine]['type']].append(self.job_status[job]['remain_process_time'])

        machine_type = self.machine_status[agent_id]['type']

        ft += [len(pt_dict[machine_type]), max(pt_dict[machine_type]), min(pt_dict[machine_type]), sum(pt_dict[machine_type])/len(pt_dict[machine_type])] \
            if len(pt_dict[machine_type]) > 0 else [0,0,0,0]

        for k in pt_dict:
            type_info = pt_dict[k]
            if len(type_info) > 0:
                ft += [len(type_info), max(type_info), min(type_info), sum(type_info)/len(type_info)]
            else:
                ft += [0,0,0,0]
        return ft

    def get_pending_jobs_feature(self, agent_id):
        # length: 40
        ft = []
        pending_time_dict = {'A':[],'B':[], 'C':[], 'D':[]}
        process_time_dict = {'A':[],'B':[], 'C':[], 'D':[]}
        for job in self.job_status:
            if self.job_status[job]['status'] == 'pending':
                op_name = self.job_status[job]['op']
                if self.job_status[job]['priority'] == 0:
                    pending_time = 99
                else:
                    pending_time = self.job_status[job]['remain_pending_time'] / self.job_status[job]['priority']
                process_time = self.job_status[job]['remain_process_time']
                machine_type = self.get_op_machine_type(op_name)
                pending_time_dict[machine_type].append(pending_time)
                process_time_dict[machine_type].append(process_time)
        machine_type = self.machine_status[agent_id]['type']
        ft += [len(pending_time_dict[machine_type]), max(pending_time_dict[machine_type]), min(pending_time_dict[machine_type]), sum(pending_time_dict[machine_type])/len(pending_time_dict[machine_type])] \
            if len(pending_time_dict[machine_type]) > 0 else [0,0,0,0]
        ft += [len(process_time_dict[machine_type]), max(process_time_dict[machine_type]), min(process_time_dict[machine_type]), sum(process_time_dict[machine_type])/len(process_time_dict[machine_type])] \
            if len(process_time_dict[machine_type]) > 0 else [0,0,0,0]

        for k in pending_time_dict:
            pending_info = pending_time_dict[k]
            process_info = process_time_dict[k]
            if len(pending_info) > 0:
                ft += [len(pending_info), max(pending_info), min(pending_info), sum(pending_info)/len(pending_info)]
                ft += [len(process_info), max(process_info), min(process_info), sum(process_info)/len(process_info)]
            else:
                ft += [0,0,0,0]
                ft += [0,0,0,0]
        return ft

    def get_op_machine_type(self, op_name):
        for type in self.env.env.job_types:
            job_type = self.env.env.job_types[type]
            for op in job_type:
                if op['op_name'] == op_name:
                    return op['machine_type']

    def get_machine_feature(self, agent_id):
        num_pending_machine = 0
        machine_type = self.machine_status[agent_id]
        for machine in self.machine_status:
            if agent_id == machine:
                continue
            else:
                if self.machine_status[machine]['type'] == machine_type:
                    if self.machine_status[machine]['status'] == 'idle':
                        num_pending_machine += 1
        return [num_pending_machine]


