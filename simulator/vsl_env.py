# import numpy as np
import math
from simulator.atsc_env import TrafficSimulator

class Vslmixedflow(TrafficSimulator):
    def __init__(self, config, port):
        super().__init__(config, port)
        self._init_sim(self.seed)
        self.terminate()

    def step(self):
        # phase = self.get_trafficlight_phase()
        # pedestrian_delay = self.get_pedestrian_delay()
        pass

    def get_MFD_sec(self, loopID1, loopID2):
        a = self.sim.inductionloop.getLastStepVehicleNumber(loopID1)
        b = self.sim.inductionloop.getLastStepVehicleNumber(loopID2)
        c = self.sim.inductionloop.getLastStepMeanSpeed(loopID1)
        d = self.sim.inductionloop.getLastStepMeanSpeed(loopID2)
        counter = 0
        speed = 0
        if c * float(a) > 0:
            speed += c * float(a)
            counter += a
        if d * float(b) > 0:
            speed += d * float(b)
            counter += b
        return counter, speed

    def get_MFD_min(self, loopID1, loopID2):
        vehiclenumsum = 0
        vehiclespeedsum = 0
        for i in range(60):
            self._simulate(1)
            vehiclenum, vehiclespeed = self.get_MFD_sec(loopID1, loopID2)
            vehiclenumsum += vehiclenum
            vehiclespeedsum = vehiclespeedsum + vehiclespeed
        average_speed = vehiclespeedsum / vehiclenumsum
        return vehiclenumsum, average_speed

    def step_get_MDF(self, loopID1, loopID2):
        buffer_num, buffer_speed = [], []
        vehicle_num, meanspeed = self.get_MFD_sec(loopID1, loopID2)
        buffer_num.append(vehicle_num)
        buffer_speed.append(meanspeed)

    def test(self, loopID1, loopID2):
        a = self.sim.inductionloop.getLastStepVehicleNumber(loopID1)
        b = self.sim.inductionloop.getLastStepVehicleNumber(loopID2)
        c = self.sim.inductionloop.getLastStepMeanSpeed(loopID1)
        d = self.sim.inductionloop.getLastStepMeanSpeed(loopID2)
        counter = 0
        counter2 = 0
        counter += a
        counter += b
        if c * float(a) > 0:
            counter2 += a
        if d * float(b) > 0:
            counter2 += b
        return counter, counter2







