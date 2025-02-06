"""
Particular class of small traffic network
@author: Tianshu Chu
"""
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import configparser
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
from envs.env import PhaseMap, PhaseSet, TrafficSimulator
from small_grid.data.build_file import gen_rou_file

sns.set_color_codes()

TEST_GRID_NEIGHBOR_MAP = {'a1': ['a2', 'a3'],
                           'a2': ['a1', 'a4'],
                           'a3': ['a1', 'a4'],
                           'a4': ['a3', 'a2']}
                           
STATE_NAMES = ['wave', 'wait']
# map from ild order (alphabeta) to signal order (clockwise from north)

NODES =  {'a1': (2, ['a2', 'a3']),
                           'a2': (2,['a1', 'a4']),
                           'a3': (2,['a1', 'a4']),
                           'a4': (2,['a3', 'a2'])}

PHASES = {2: ['GGgrrrGGgrrr', 'rrrGGgrrrGGg']}
class TestGridPhase(PhaseMap):
    def __init__(self):
        two_phase = ['GGgrrrGGgrrr', 'rrrGGgrrrGGg']
        self.phases = {2: PhaseSet(two_phase)}


class  TestGridController:
    def __init__(self, node_names, nodes):
        self.name = 'greedy'
        self.node_names = node_names
        self.nodes = nodes

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # get the action space
        phases = PHASES[NODES[node_name][0]]
        flows = []
        node = self.nodes[node_name]
        # get the green waves
        for phase in phases:
            wave = 0
            visited_ilds = set() 
            for i, signal in enumerate(phase):
                if signal == 'G':
                    # find controlled lane
                    lane = node.lanes_in[i]
                    # ild = 'ild:' + lane
                    ild = lane
                    # if it has not been counted, add the wave
                    if ild not in visited_ilds:
                        j = node.ilds_in.index(ild)
                        wave += ob[j]
                        visited_ilds.add(ild)
            flows.append(wave)
        return np.argmax(np.array(flows))


class TestGridEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.num_car_hourly = config.getint('num_extra_car_per_hour')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_node_phase_id(self, node_name):
        if node_name == 'nt1':
            return 3
        return 2

    def _init_map(self):
        self.neighbor_map = TEST_GRID_NEIGHBOR_MAP
        self.phase_map = TestGridPhase()
        self.state_names = STATE_NAMES

    def _init_sim_config(self, seed):
        return "/home/taekwon/projects/deeprl_signal_control/test_grid/data/test_grid.sumocfg"

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_greedy_test.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = TestGridEnv(config['ENV_CONFIG'], 2, base_dir, is_record=True, record_stat=True)
    ob = env.reset()
    controller = TestGridController(env.node_names, env.nodes)
    rewards = []
    it = 0
    while True and it <20:
        it += 1
        next_ob, _, done, reward = env.step(controller.forward(ob))
        rewards.append(reward)
        if done:
            break
        ob = next_ob
    env.plot_stat(np.array(rewards))
    logging.info('avg reward: %.2f' % np.mean(rewards))
    env.terminate()
    time.sleep(2)
    env.collect_tripinfo()
    env.output_data()
