"""

仿真控制文件、控制sumo进行仿真

"""

import numpy as np
from simulator.atsc_env import TrafficSimulator


class Expressway(TrafficSimulator):
    def __init__(self, config, port):
        super().__init__(config, port)
        self._init_sim(self.seed)
        self.name = config.get('scenario')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        self.action_space = 8
        self.state_space = len(self._get_state())
        self.prev_action = [-1, -1, -1]
        self.global_reward = 0
        self._init_map()
        self._init_nodes()
        self.terminate()

    # 进行一步仿真
    def step(self, action):
        phaselist = self.set_action_phase(action)
        # pre_state = self._get_state()
        for i in range(len(action)):
            if self.prev_action[i] == action[i]:
                self.sim.trafficlight.setRedYellowGreenState(self.tl_nodes[i], phaselist[i])
                self.sim.trafficlight.setPhaseDuration(self.tl_nodes[i], self.yellow_interval_sec)
            else:
                self.sim.trafficlight.setRedYellowGreenState(self.tl_nodes[i], 'yGGG')
                self.sim.trafficlight.setPhaseDuration(self.tl_nodes[i], self.yellow_interval_sec)
        self._simulate(self.yellow_interval_sec)
        rest = self.control_interval_sec - self.yellow_interval_sec
        for i in range(len(action)):
            self.sim.trafficlight.setRedYellowGreenState(self.tl_nodes[i], phaselist[i])
            self.sim.trafficlight.setPhaseDuration(self.tl_nodes[i], rest)
        self._simulate(rest)
        self.prev_action = action
        state = self._get_state()
        halting = self._get_halting()
        flowout = self._get_flowout()
        # reward = -np.sum(state)-pow(np.sum(n), 2)
        reward = np.sum(flowout)
        done = False
        if self.cur_sec >= self.episode_length_sec:
            done = True
        self.global_reward = self.global_reward + reward
        return state, reward, done, self.global_reward

    # 调用traci获取的各节点状态信息，生成系统状态信息
    def _get_state(self):
        state = []
        a = ['%d' % i for i in range(1, 14)]
        for i, lane in enumerate(a):
            state.append(self.sim.edge.getLastStepVehicleNumber(lane))
        ab = ['%d' % i for i in range(15, 21)]
        for i, lane_in in enumerate(ab):
            state.append(self.sim.edge.getLastStepVehicleNumber(lane_in))
        return state

    #  系统流出车流量
    def _get_flowout(self):
        flowout = []
        flowout.append(self.sim.edge.getLastStepVehicleNumber('13'))
        flowout.append(self.sim.edge.getLastStepVehicleNumber('16'))
        flowout.append(self.sim.edge.getLastStepVehicleNumber('18'))
        flowout.append(self.sim.edge.getLastStepVehicleNumber('20'))
        return flowout

    #  匝道排队数量
    def _get_halting(self):
        halting = []
        halting.append(self.sim.edge.getLastStepVehicleNumber('15'))
        halting.append(self.sim.edge.getLastStepVehicleNumber('17'))
        halting.append(self.sim.edge.getLastStepVehicleNumber('19'))
        return halting

    # 导入节点名称信息
    def _init_map(self):
        self.node_names = ['0%d' % i for i in range(1, 21)]
        self.n_node = 20

    # 获取信号灯节点名称
    def _init_nodes(self):
        tl_nodes = self.sim.trafficlight.getIDList()
        self.tl_nodes = tl_nodes

    # 获取action对应相位设定字符串组
    @staticmethod
    def set_action_phase(action):
        action_phase = []
        for i in range(len(action)):
            if action[i] == 0:
                action_phase.append('rGGG')
            else:
                action_phase.append('GGGG')
        return action_phase


'''
action为三维数组，每个元素对应一个匝道，取0为匝道关闭，取1为匝道开放，共8种action
'''
