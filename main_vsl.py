import configparser
from simulator.vsl_env import Vslmixedflow
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config/config_VSLmixedflow.ini')
    env = Vslmixedflow(config['ENV_CONFIG'], 7800)
    state_0 = env.reset()
    x = []
    y = []
    vehiclenumsum = 0
    vehiclespeedsum = 0
    # countersum = 0
    for i in range(5100):
        env._simulate(1)
        vehiclenum, vehiclespeed = env.get_MFD_sec("det_L0_0", "det_L0_1")
        vehiclenumsum += vehiclenum
        vehiclespeedsum = vehiclespeedsum + vehiclespeed
        if (i % 300 == 0) and (i > 0):
            meanspeed = vehiclespeedsum / vehiclenumsum
            print(vehiclenumsum, meanspeed)
            occ = env.sim.edge.getLastStepOccupancy('L0')
            print('occ = ', occ)
            vehiclenumsum = 0
            vehiclespeedsum = 0

