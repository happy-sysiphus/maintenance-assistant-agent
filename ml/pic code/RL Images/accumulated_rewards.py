import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

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
print(len(data))

sum1 = list()
temp = 0.0
for i in range(30):
    temp = temp + data[0][i] + data[6][i] - data[12][i] - data[18][i]
    sum1.append(temp)

sum2 = list()
temp = 0.0
for i in range(30):
    temp = temp + data[1][i] + data[7][i] - data[13][i] - data[19][i]
    sum2.append(temp)

sum3 = list()
temp = 0.0
for i in range(30):
    temp = temp + data[2][i] + data[8][i] - data[14][i] - data[20][i]
    sum3.append(temp)

sum4 = list()
temp = 0.0
for i in range(30):
    temp = temp + data[3][i] + data[9][i] - data[15][i] - data[21][i]
    sum4.append(temp)

sum5 = list()
temp = 0.0
for i in range(30):
    temp = temp + data[4][i] + data[10][i] - data[16][i] - data[22][i]
    sum5.append(temp)

sum6 = list()
temp = 0.0
for i in range(30):
    temp = temp + data[5][i] + data[11][i] - data[17][i] - data[23][i]
    sum6.append(temp)


sum_list = [sum1, sum2, sum3, sum4, sum5, sum6]
days = [i for i in range(1,31,1)]

methodlist = ['inv RL', 'epsilon-0.3','epsilon-0.7', 'epsilon-1',  'stNo-456', 'p-0.5']
colorList = ["#3498db", "#9b59b6", "k", "r", "orange", "#2ecc71"] # darkviolet
markerList = ['o', '+', 'h', '^', '*', 's', 'h', '2', 'p']


def smooth_xy(lx, ly):
    """数据平滑处理

    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 30)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth, y_smooth


if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(14,8))

    for index in range(len(methodlist)):

        x, y = smooth_xy(days, sum_list[index]) # 光滑处理

        #x_scatter = np.linspace(0, x.max(), 7) *2
        ax.plot(x
                , y
                , linewidth=4
                , color=colorList[index]  # 线的颜色
                , marker=markerList[index]  # marker形状
                , ms=10  # marker size
                , mec=colorList[index]  # marker边框颜色
                , linestyle='-'
                #, mfc=colorList[index]  # marker实心颜色
                , mfc='white'  # marker实心颜色
                , label=methodlist[index]  # legend显示
                #, clip_on=False
                )


   # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')  # 科学计数法
    ax.set_xlabel("Days", fontsize=32, fontproperties='Times New Roman'
                  #, fontweight='bold'
                  )
    ax.set_ylabel('Rewards', fontsize=32, fontproperties='Times New Roman'
                  #, fontweight='bold'
                  #, rotation = 0
                  #, labelpad=15
                  )


    #ax.patch.set_facecolor((0.925, 0.125, 0.90, 0.035))
    xticks = [1, 5, 10, 15, 20, 25, 30]
    plt.xticks(xticks, fontproperties='Times New Roman', fontsize=20)
    plt.yticks(fontproperties='Times New Roman', fontsize=20)
    plt.grid(True, linestyle=':',alpha=0.6)
    plt.legend(prop={'size': 32, 'family': 'Times New Roman'}, loc='upper left', framealpha=0.6)
    fig.tight_layout()
    plt.savefig('./pic/accumula_rewards.png'
                #, format='pdf'
                , bbox_inches='tight'
                , pad_inches=0.1, dpi=500)
    plt.show()



