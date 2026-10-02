import numpy as np
import matplotlib.pyplot as plt

day_num = 30

match_pro1 = 30
match_pro2 = 1.5
match_pro3 = 0.2
self_cost = -1.5
inv_cost = 2

colorList = [ "#3498db", "#9b59b6","k", "r", "orange", "#2ecc71"] # darkviolet
markerList = ['o', '+', 'h', '^', '*', 's', 'h', '2', 'p']

file_name = "AllData.txt"
data = list()
with open(file_name, 'r') as f:
    for line in f:
        data_ = line.strip().split(', ')
        data.append([float(data_str) for data_str in data_])
data = data[18: 24]
print(len(data))

if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(18,8))

    days = [i for i in range(1, 31, 1)]
    # '''折线图'''
    # plt.plot(days, data[0], 'b-', label="RL")
    # plt.plot(days, data[1], 'r-', label="epsilon=0.3")
    # plt.plot(days, data[2], 'g-', label="epsilon=0.7")
    # plt.plot(days, data[3], 'b--', label="epsilon=1")
    # plt.plot(days, data[4], 'r--', label="stNo=456")
    # plt.plot(days, data[5], 'g--', label="p=0.5")

    '''堆积柱状图'''
    bottom_y = [0] * len(data[0])  # # 将bottom_y元素都初始化为0
    width = 0.8
    for index, y in enumerate(data):

        plt.bar(days, y, width, bottom=bottom_y, color=colorList[index])
        # 累加数据计算新的bottom_y
        bottom_y = [a + b for a, b in zip(y, bottom_y)]

    ax.set_xlabel("Days", fontsize=32, fontproperties='Times New Roman'
                  #, fontweight='bold'
                  )
    ax.set_ylabel("Weight punishment", fontsize=32, fontproperties='Times New Roman'
                  #, fontweight='bold'
                  )
    #ax.patch.set_facecolor((0.925, 0.125, 0.90, 0.015))
    xticks = [1, 5, 10, 15, 20, 25, 30]
    plt.xticks(xticks, fontproperties='Times New Roman', fontsize=20)
    plt.yticks(fontproperties='Times New Roman', fontsize=20)
    plt.legend([ 'RL', 'epsilon-0.3','epsilon-0.7', 'epsilon-1',  'stNo-456', 'p-0.5'], prop={'size': 32, 'family': 'Times New Roman'},)
    fig.tight_layout()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('./pic/bar_weight_punishment.png'
                # , format='pdf'
                , bbox_inches='tight'
                , pad_inches=0.1, dpi=500)

    plt.show()




