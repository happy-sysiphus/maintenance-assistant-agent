import numpy as np
import matplotlib.pyplot as plt
# RL，全部给库存，0.8给库存，完全greedy，剩余的456给库存，0.5的参与匹配
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
data = data[18: 24]
print(len(data))

# days = [i for i in range(30)]
# plt.plot(days, R2, 'b--', label="inv RL")
# plt.plot(days, R2_s1, 'r--', label="inv cplex1")
# plt.plot(days, R2_s11, 'g--', label="inv cplex2")
# plt.xlabel("days")
# plt.ylabel("inventory cost")
# plt.title("inv-day")
# plt.legend()
# plt.show()
plt.figure(figsize=(20,8),dpi=80)
total_wid = 0
for day_j in range(day_num):

    # 归一化，并且当某一项为0时，给予0.1（图中显示会更舒服一些）
    max_val = max(map(max, data))
    r1 = data[0][day_j] / max_val if data[0][day_j] != 0 else 0.01
    r2 = data[1][day_j] / max_val if data[1][day_j] != 0 else 0.01
    r3 = data[2][day_j] / max_val if data[2][day_j] != 0 else 0.01
    r4 = data[3][day_j] / max_val if data[3][day_j] != 0 else 0.01
    r5 = data[4][day_j] / max_val if data[4][day_j] != 0 else 0.01
    r6 = data[5][day_j] / max_val if data[5][day_j] != 0 else 0.01

    plt.barh(1, r1, height=1, left=day_j+1, color='g', edgecolor='black')
    plt.barh(2, r2, height=1, left=day_j+1, color='r', edgecolor='black')
    plt.barh(3, r3, height=1, left=day_j + 1, color='b', edgecolor='black')
    plt.barh(4, r4, height=1, left=day_j + 1, color='g', edgecolor='black')
    plt.barh(5, r5, height=1, left=day_j + 1, color='r', edgecolor='black')
    plt.barh(6, r6, height=1, left=day_j + 1, color='b', edgecolor='black')
    #plt.barh(3, R2_s11[day_j] + 1, height=0.5, left=day_j+1, color='g', edgecolor='black')
    #total_wid += max(R2[day_j], R2_s1[day_j], R2_s11[day_j]) + 5

plt.grid(axis='x',which='major')
plt.ylabel("strategies")
plt.xticks(range(1, 31,1))
plt.yticks(range(7), ["", "RL", "epsilon=03", "epsilon=0.7", "epsilon=1", "stNo=456", "p=0.5"], rotation=90)
plt.title("weight penalty-days")
plt.show()


