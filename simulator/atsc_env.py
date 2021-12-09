"""

仿真控制文件、控制sumo进行仿真

"""
import subprocess
from sumolib import checkBinary
import time
import traci
import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class TrafficSimulator:
    def __init__(self, config, port):
        self.seed = config.getint('seed')
        self.sumocfg = config.get('sumocfg')
        self.cur_sec = 0
        self.cur_episode = 0
        self.port = port
        self.gui = False

    # 连接sumo仿真系统，仿真文件位置需要手动输入
    def _init_sim(self, seed):
        sumocfg_file = self.sumocfg
        if self.gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', sumocfg_file]
        # command += ['--random']
        command += ['--seed', str(seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        command += ['--time-to-teleport', '600']  # long teleport for safety
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        subprocess.Popen(command)
        # wait 1s to establish the traci server
        time.sleep(1)
        self.sim = traci.connect(port=self.port)

    # 仿真环境初始化，需先运行terminate()
    def reset(self, gui=False):
        seed = self.seed
        self._init_sim(seed)
        self.cur_sec = 0
        self.cur_episode += 1
        self.seed += 1
        return self._get_state()

    # 进行num_step个simulationStep
    def _simulate(self, num_step):
        for _ in range(num_step):
            self.sim.simulationStep()
            self.cur_sec += 1

    # 仿真终止
    def terminate(self):
        self.sim.close()

    # 调用traci获取的各节点状态信息，生成系统状态信息
    def _get_state(self):
        # needs to overwrite
        pass

