"""
Loading sumo network data

include : node edge connection tls phase

@author: wdz
"""
from sumolib import checkBinary
import os
import sys
import traci
import subprocess
import time
import logging


class Node:
    def __init__(self):
        self.node = []
        self.port = 8000
        # self._init_sim(12)

    def get_node(self):
        node = ['nt1', 'nt2', 'nt3']
        # 试图从路网文件中提取节点名，目前无法实现，需要手动输入节点名
        self.node = node


class Phase(Node):
    def __init__(self):
        super(Phase, self).__init__()

    def get_node_phase(self):
        node_id = self.node[1]
        phase = self.sim.trafficlight.getControlledLanes(node_id)
        return phase


    # def get_phase(self):
    #     self.node[i].phase = traci.getphase


net_file = "../sumo_file/design_grid/exp.net.xml"
net_data = open(net_file, "r")
net_data.seek(0, 0)
a = net_data.read(60)
net_data.close()
print(a)

# node = Phase()
# node.get_node()
# b = node.get_node_phase()
#
# print(b)