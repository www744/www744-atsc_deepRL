import numpy as np
import cvxopt
from simulator.atsc_env import TrafficSimulator


class Offlinephase(TrafficSimulator):
    def __init__(self, config, port, ped_rate, veh_rate, saturation, crossing_time):
        super().__init__(config, port)
        self.ped_rate = ped_rate
        self.crossing_time = crossing_time
        self.veh_rate = veh_rate
        self.saturation = saturation
        self._init_sim(self.seed)
        self.terminate()

    '''二次项系数，南北直行、南北左转、东西直行、东西左转'''
    '''行人流量，南北方向过街，东西方向过街'''
    def get_ped_delay_index(self):
        delay_index_1 = np.mat([0, 1, 1, 1, self.crossing_time[0]])
        delay1 = delay_index_1.T * delay_index_1
        delay_index_2 = np.mat([1, 1, 0, 1, self.crossing_time[1]])
        delay2 = delay_index_2.T * delay_index_2
        delay = self.ped_rate[0] * delay1 + self.ped_rate[1] * delay2
        p_ped = delay[:4, :4]
        q_ped = delay[4, :4]
        return p_ped, q_ped

    def get_veh_delay_index(self):
        index_up = np.multiply(self.veh_rate, self.saturation)/3600/3600
        index_down = (self.saturation - self.veh_rate) / 3600
        index = np.multiply(index_up, 1 / index_down)
        index = np.array(index)
        p_veh = np.mat(np.zeros((4, 4)))
        for i in range(4):
            a = [1, 1, 1, 1]
            a[i] = 0
            veh_delay = np.mat(a)
            veh_delay_squ = veh_delay.T * veh_delay
            delay = index[0][i] * veh_delay_squ
            p_veh = p_veh + delay
        return p_veh

    '''
    约束条件的参数矩阵需要在该函数中直接修改
    目前的约束条件包括最短周期、最长周期、各相位最短时间
    '''
    def get_offline_phase(self):
        p_ped, q_ped = self.get_ped_delay_index()
        p_veh = self.get_veh_delay_index()
        p = cvxopt.matrix(p_ped + p_veh)
        q = cvxopt.matrix(q_ped.T)
        g = np.mat([[1.0, 1.0, 1.0, 1.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0], [-1.0, -1.0, -1.0, -1.0]])
        g = cvxopt.matrix(g)
        h = cvxopt.matrix([120.0, -15.0, -5.0, -15.0, -5.0, -60.0])
        result = cvxopt.solvers.qp(p, q, g, h)
        phase_time = np.array(result['x']).astype(int)
        return phase_time

    def set_trafficlight_program(self):
        phase_time = self.get_offline_phase()
        tls = self.sim.trafficlight.getAllProgramLogics('center')
        self.sim.trafficlight.setProgramLogic('center', tls)
