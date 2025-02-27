from envs.env import PhaseMap, PhaseSet, TrafficSimulator
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import configparser
import os
import sys
import subprocess
from worli.data.create_trip import main as create_trip
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

matplotlib.use('Agg')

sns.set_color_codes()

STATE_NAMES = ['wave', 'wait']

NODES = {
    "node1": ("1.1", ["node2", "node3", "node6"]),
    "node2": ("1.2", ["node1", "node4"]),
    "node3": ("1.3", ["node1", "node4", "node6"]),
    "node4": ("1.4", ["node2", "node3", "node5", "node6"]),
    "node5": ("1.5", ["node4", "node7"]),
    "node6": ("1.6", ["node1", "node3", "node4", "node7"]),
    "node7": ("1.7", ["node5", "node6"]),
}

PHASES = {
    "1.1": ["rGGrrrrrrrrGGrr", "ryyrrrrrrrryyrr", "GrrgGrrrrrGrrgG", "yrrgyrrrrryrryy", "rrrGrGGGGgrrrrr", "rrryryyyyyrrrrr"],  # Node1
    "1.2": ["GGG", "yyy", "rrr"],  # Node2
    "1.3": ["GGG", "yyy", "rrr"],  # Node3
    "1.4": ["rGGgGGrrrrrrr", "ryyyyyrrrrrrr", "GrrrrrgGrrrrr", "yrrrrrgyrrrrr", "rrrrrrGrGGggG", "rrrrrryryyyyy"],  # Node4
    "1.5": [
        "rrrrrrrrrrrrrrrGGrrrrrrrrrrrGGrrr",
        "rrrrrrrrrrrrrrryyrrrrrrrrrrryyrrr",
        "rrrrrrrrrrrrrGGrrggGrrrrrrGGrrggG",
        "rrrrrrrrrrrrryyrryyyrrrrrryyrryyy",
        "rrGGrrrrrrrrrrrrrrrrrrGrrrrrrrrrr",
        "rryyrrrrrrrrrrrrrrrrrryrrrrrrrrrr",
        "GGrrggGrrrrrrrrrrrrrGGrggGrrrrrrr",
        "yyrryyyrrrrrrrrrrrrryyryyyrrrrrrr",
        "rrrrrrrGGGGGGrrrrrrrrrrrrrrrrrrrr",
        "rrrrrrryyyyyyrrrrrrrrrrrrrrrrrrrr"
    ],  # Node5
    "1.6": ["GGggrrrGGGg", "yyyyrrrGyyy", "rrrrGGgGrrr", "rrrryyyGrrr"],  # Node6
    "1.7": ["GggrrrGGg", "yyyrrrGyy", "rrrGGgGrr", "rrryyyGrr"],  # Node7
}

SECONDARY_ILDS_MAP = {
    '658655412#8_0': '1201591783_0',
    '-1133675674#0_0': '-1133675674#1_0',
    '1237761220_0': '-1237761220_0',
    '1237761225#1_0': '1237761220_0',
    '1285590644#1_0': '1285590644#0_0',
    '1133675674#3_0': '1133675674#2_0',
    '-1133675674#1_0': '-1133675674#2_0',
    '658655411#10_0': '658655411#9_0',
    '-1133675674#2_0': '-1133675674#3_0',
    '658655411#8_0': '658655411#7_0',
    '1234788292#0_0': '1236044674#1_0',
    '1237761221_0': '-1237761221_0',
    '658655412#3_0': '658655412#2_0',
    '1236044643#1_0': '1236044674#1_0',
    '658655411#2_0': '658655412#4_0',
    '-610651679#2_0': '610651679#1_0',
    '-1133675674#4_0': '-1133675674#5_0',
    '658655411#5_0': '658655411#4_0',
    '1133675674#2_0': '1133675674#1_0',
    '658655412#0_0': '658655411#11_0',
    '-1237761225#1_0': '-610651679#0_0',
    '610651679#1_0': '610651679#0_0',
    '1133675674#6_0': '-1133675674#6_0',
    '658655412#2_0': '658655411#6_0',
    '-1133675674#5_0': '-1133675674#6_0',
    '658655411#7_0': '658655411#6_0',
    '-610651679#0_0': '610651679#0_0',
    '658655411#0_0': '1249027317_0',
    '658655411#9_0': '658655411#8_0',
    '1237761233#1_0': '-1237761233#1_0',
    '-1133675674#6_0': '-1237761220_0',
    '658655412#5_0': '658655412#4_0',
    '-1237761221_0': '61651851_0',
    '-1237761220_0': '1062054751_0',
    '61651851_0': '-610651679#0_0',
    '658655411#6_0': '658655412#2_0',
    '1133675674#5_0': '1133675674#4_0',
    '1285590647#0_0': '1234788292#2_0',
    '1285590647#1_0': '1234788292#1_0',
    '1133675674#0_0': '1237761233#1_0',
    '658655412#1_0': '658655412#0_0',
    '1285590644#0_0': '1236044674#1_0',
    '1133675674#4_0': '1133675674#3_0',
    '658655411#3_0': '658655411#2_0',
    '1062054751_0': '61651851_0',
    '1236044674#1_0': '1236044674#0_0',
    '658655411#1_0': '658655412#6_0',
    '1201591783_0': '658655412#8_0',
    '658655411#11_0': '658655411#10_0',
    '-1237761233#1_0': '1237761233#1_0',
    '1234788292#2_0': '1234788292#1_0',
    '658655412#7_0': '658655412#6_0',
    '1177154782_0': '658655411#11_0',
    '658655411#4_0': '658655411#3_0',
    '1123855695_0': '658655412#8_0',
    '1133675674#1_0': '1133675674#0_0',
    '658655412#6_0': '658655412#5_0',
    '1234788292#1_0': '1234788292#0_0',
    '658655412#4_0': '658655411#3_0',
    '610651679#0_0': '-610651679#0_0',
    '1249027317_0': '-1237761233#1_0',
    '-1133675674#3_0': '-1133675674#4_0'
}

class WorliPhase(PhaseMap):
    def __init__(self):
        self.phases = {}
        for key, val in PHASES.items():
            self.phases[key] = PhaseSet(val)


class WorliController:
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


class WorliEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.num_car_hourly = config.getint('num_extra_car_per_hour')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_node_phase_id(self, node_name):
        return self.phase_node_map[node_name]

    def _init_neighbor_map(self):
        return dict([(key, val[1]) for key, val in NODES.items()])

    def _init_map(self):
        self.neighbor_map = self._init_neighbor_map()
        self.phase_map = WorliPhase()
        self.phase_node_map = dict([(key, val[0])
                                   for key, val in NODES.items()])
        self.state_names = STATE_NAMES
        self.secondary_ilds_map = SECONDARY_ILDS_MAP

    def _init_sim_config(self, seed=None):
        if self.cur_episode == 0:
            create_trip()

        return os.path.join(os.getcwd(), 'worli', 'data', 'worli.sumocfg')

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(os.path.join(self.output_path, self.name + '_' + name + '.png'))


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_greedy_worli.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = WorliEnv(config['ENV_CONFIG'], 2, base_dir,
                   is_record=True, record_stat=True)
    ob = env.reset()
    controller = WorliController(env.node_names, env.nodes)
    rewards = []
    it = 0
    while True:
        it += 1
        next_ob, _, done, reward = env.step(controller.forward(ob))
        rewards.append(reward)
        if done:
            break
        ob = next_ob
    # while True:
    #     next_ob, _, done, reward = env.baseCaseStep()
    #     rewards.append(reward)
    #     if done:
    #         break
    env.plot_stat(np.array(rewards))
    logging.info('avg reward: %.2f' % np.mean(rewards))
    env.terminate()
    time.sleep(2)
    env.collect_tripinfo()
    env.output_data()
