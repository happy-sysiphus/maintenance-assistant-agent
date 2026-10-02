import numpy as np
import matplotlib.pyplot as plt

day_num = 30

match_pro1 = 30
match_pro2 = 1.5
match_pro3 = 0.2
self_cost = -1.5
inv_cost = 2

file_name = "AllData.txt"
data = list()
with open(file_name, 'r') as f:
    for line in f:
        data_ = line.strip().split(', ')
        data.append([float(data_str) for data_str in data_])

data = data[6: 12]
print(len(data))

days = [i for i in range(30)]
plt.plot(days, data[0], 'b-', label="RL")
plt.plot(days, data[1], 'r-', label="epsilon=0.3")
plt.plot(days, data[2], 'g-', label="epsilon=0.7")
plt.plot(days, data[3], 'b--', label="epsilon=1")
plt.plot(days, data[4], 'r--', label="stNo=456")
plt.plot(days, data[5], 'g--', label="p=0.5")
plt.xlabel("days")
plt.ylabel("self design")
plt.title("self design-day")
plt.legend()
plt.show()



