import numpy as np


time_internal = np.random.exponential(scale=2, size=3600)
print('t', time_internal)
depart_time = []
depart_time.append(0)
for i in range(3600):
    sum = time_internal[i] + depart_time[i]
    depart_time.append(sum)
print('d', depart_time)
