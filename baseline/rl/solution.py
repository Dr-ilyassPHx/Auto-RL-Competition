from train_env import get_env, MaskedActionsModel
from agent import Agent
import ray
import time
import random
from collections import defaultdict

from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer

ray.init()


class Trainer:
    def __init__(self, Env, conf_list):
        self.conf_list = conf_list
        self.Env = Env
        self.checkpoint = None
        self.iter = 0
        ModelCatalog.register_custom_model('MaskModel', MaskedActionsModel)

    def train(self, run_time):
        rl_env = get_env(self.Env, self.conf_list)
        trainer = PPOTrainer(env=rl_env, config={
            'train_batch_size':20000,
            'num_workers':6,
            'num_gpus':1,
            'sgd_minibatch_size':2048,
            'model':{
                'custom_model': 'MaskModel'
            },
        })
        now_time = time.time()
        total_time = 0
        while True:
            last_time = now_time

            result = trainer.train()
            reward = result['episode_reward_mean']
            print(f'Iteration: {self.iter}, reward: {reward}, training iteration: {trainer._iteration}')
            now_time = time.time()
            total_time += now_time - last_time
            trainer.save(f'./work')
            self.checkpoint = f'./work/checkpoint_{trainer._iteration}/checkpoint-{trainer._iteration}'
            self.iter += 1

            if total_time + 2*(now_time-last_time) > run_time:
                break


        return Agent(trainer, rl_env)





