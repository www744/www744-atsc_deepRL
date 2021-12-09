import configparser
from simulator.isolate_env import Isolate
import simulator.isolate_offline
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config/config_isolate_binomial.ini')
    rate = np.mat([400, 300, 1050, 620])
    saturation = np.mat([3537, 3789, 1965, 2105])
    ped_rate = [1620 / 3600, 1080 / 3600]  # 单位： ped/s
    crossing_time = [12, 12]  # 单位： s
    env = Isolate(config['ENV_CONFIG'], 7800)
    b = env.reset()
    all_delay = []
    for i in range(3600):
        env._simulate(1)
        all_delay.append(env.get_delay())
    x = np.arange(0, 3600)
    a = sum(all_delay)/3600
    '''
    uniform = 73.52   binomial = 72.40
    '''
    print(a)
    plt.title("binomial delay")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    plt.plot(x, all_delay)
    plt.show()
