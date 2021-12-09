import numpy as np
import math
from simulator.atsc_env import TrafficSimulator


class Isolate(TrafficSimulator):
    def __init__(self, config, port):
        super().__init__(config, port)
        self._init_sim(self.seed)
        self.terminate()

    def step(self):
        # phase = self.get_trafficlight_phase()
        # pedestrian_delay = self.get_pedestrian_delay()
        pass

    def _get_state(self):
        pass

    '''
    获得三个方向的行人过街需求估计，南北直行、东西直行、对角过街，
    相关参数，南北，东西，对角，不过街的行人比例
    '''
    def get_pedestrian_delay(self, walkingarea, edge_id0):
        waiting_time = self._get_pedestrian_waiting_time(walkingarea)
        arriving = self._get_pedestrian_arriving_time(edge_id0)
        print('arriving = ', arriving)
        t = np.zeros((len(waiting_time), 11))
        for i in range(11):
            if i == 0:
                for j in range(len(waiting_time)):
                    t[j][i] = waiting_time[j]
            else:
                t[0][i] = arriving[i - 1] * 0.8 * 0.4
                t[1][i] = arriving[i - 1] * 0.8 * 0.5
                t[2][i] = arriving[i - 1] * 0.8 * 0.1
        return t

    '''
    获取行人等待区相关数据，等待区id为 ：center_w0(左上)，w1(右上), w2(右下), w3(左下)
    目前没考虑上一决策时刻的情况，即确定性的过街需求目前还没有考虑
    '''
    def _get_pedestrian_waiting_time(self, walkingarea):
        pedestrian_idlist = self.sim.edge.getLastStepPersonIDs(walkingarea)
        pedestrian_num = len(pedestrian_idlist)
        delay = [0, 0, 0]
        delay[0] = pedestrian_num * 0.4
        delay[1] = pedestrian_num * 0.5
        delay[2] = pedestrian_num * 0.1
        return delay

    '''
    获得行人到达等待区的时间和人数，需要对8条边均进行计算，同时要去除远离等待区的行人
    '''
    def _get_pedestrian_arriving_time(self, edge_id):
        pedestrian_idlist = self.sim.edge.getLastStepPersonIDs(edge_id)
        pedestrian_num = len(pedestrian_idlist)
        print('pedestrian_num = ', pedestrian_num)
        t = np.zeros(10)
        pedestrian_info = [[0 for col in range(5)] for row in range(pedestrian_num)]
        if edge_id == ('n2c' or 'c2n'):
            position_probe = 2
            angle_probe = 180
        elif edge_id == ('s2c' or 'c2s'):
            position_probe = 2
            angle_probe = 0
        elif edge_id == ('w2c' or 'c2w'):
            position_probe = 1
            angle_probe = 90
        else:
            position_probe = 1
            angle_probe = 270
        for j in range(pedestrian_num):
            pedestrian_info[j][0] = pedestrian_idlist[j]
            position = self.sim.person.getPosition(pedestrian_idlist[j])
            pedestrian_info[j][1] = position[0]
            pedestrian_info[j][2] = position[1]
            pedestrian_info[j][3] = math.ceil(((abs(pedestrian_info[j][position_probe]) - 13.6) / 1.5))
            pedestrian_info[j][4] = self.sim.person.getAngle(pedestrian_idlist[j])
            if pedestrian_info[j][3] <= 10 and pedestrian_info[j][4] == angle_probe:
                t[pedestrian_info[j][3]-1] += 1
        return t

    '''
    基于到达时间，计算t时刻该入口方向，不同行驶方向的延误时间，目前假设左至右比例为0.4，0.4，0.2，不同比例延误估计有差异
    顺序为 右转、直行、左转
    '''
    def get_vehicle_delay(self, edge_id):
        delay = np.zeros((3, 10))
        t = self._get_vehicle_arriving_time(edge_id)
        probility = (0.4, 0.4, 0.2)
        probility_right = 2 * probility[2]/(1 - probility[0])
        for right in range(10):
            if right == 0:
                delay[0][right] = round(t[0][0] * probility_right, 1)
            else:
                delay[0][right] = round(delay[0][right - 1] + t[0][right] * probility_right, 1)
        for straight in range(10):
            if straight == 0:
                delay[1][straight] = round(t[0][0] * (1 - probility_right) + t[1][0], 1)
            else:
                delay[1][straight] = round(delay[1][straight-1] + t[0][straight]*(1-probility_right) + t[1][straight], 1)
        for left in range(10):
            if left == 0:
                delay[2][left] = t[2][0]
            else:
                delay[2][left] = delay[2][left - 1] + t[2][left]
        return delay

    '''
    车辆到达时间获取，t为3*10矩阵，代表每个车道t秒后进入排队的车辆数，t0为当前排队车辆数，t可用来估计一段时间内的机动车延误
    1是直右车道，2是直行车道，3是左转车道
    '''
    def _get_vehicle_arriving_time(self, edge_id):
        lane_num = self.sim.edge.getLaneNumber(edge_id)
        t = np.zeros((3, 10))
        for i in range(1, lane_num):
            lane_vehicle_idlist = self.sim.lane.getLastStepVehicleIDs(edge_id + str('_') + str(i))
            vehicle_num = len(lane_vehicle_idlist)
            lane_vehicle_info = [[0 for col in range(3)] for row in range(vehicle_num)]
            for j in range(vehicle_num):
                lane_vehicle_info[j][0] = lane_vehicle_idlist[j]
                position = self.sim.vehicle.getPosition(lane_vehicle_idlist[j])
                lane_vehicle_info[j][1] = position[0]
                lane_vehicle_info[j][2] = position[1]
            halting_num = self.sim.lane.getLastStepHaltingNumber(edge_id + str('_') + str(i))
            t[i-1][0] = halting_num
            if edge_id == 'n2c' or 's2c':
                position_probe = 2
                print('s = ', position_probe)
            else:
                position_probe = 1
            if halting_num == 0:
                queue_position = 13.6
            else:
                queue_position = lane_vehicle_info[vehicle_num - halting_num][position_probe]
            for num in range(vehicle_num - halting_num):
                arriving_time = math.ceil((abs(lane_vehicle_info[num][position_probe]) - queue_position - 7.5 *
                                              (vehicle_num - halting_num - num)) / 10)
                t[i - 1][arriving_time] = t[i - 1][arriving_time] + 1
        return t

    def get_trafficlight_phase(self):
        phase = self.sim.trafficlight.getRedYellowGreenState('center')
        program = self.sim.trafficlight.getAllProgramLogics('center')
        print('program')
        print(program)
        return phase

    def set_trafficlight_phase_or(self, phase, duration):
        self.sim.trafficlight.setPhase('center', phase)
        self.sim.trafficlight.setPhaseDuration('center', duration)

    def get_best_phase_duration(self, current_phase, pedestrian_delay, vehicle_delay):

        pass

    def get_best_offline_phase(self):
        pedestrian_delay = 0
        pass

    def get_delay(self):
        pedestrian_num = 0
        veh_num = 0
        edgeID = ['n2c', 's2c', 'e2c', 'w2c']
        for i in range(4):
            pedestrian_idlist = self.sim.edge.getLastStepPersonIDs(':center_w' + str(i))
            pedestrian_num = pedestrian_num + len(pedestrian_idlist)
            veh_halting = self.sim.edge.getLastStepHaltingNumber(edgeID[i])
            veh_num = veh_num + veh_halting
        all_delay = pedestrian_num + veh_num
        return all_delay













