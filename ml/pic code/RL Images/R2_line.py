import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import seaborn as sns; sns.set()
sns.set_style("white")

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
data = data[12: 18]
print(len(data))


methodlist = ['inv RL', 'epsilon-0.3','epsilon-0.7', 'epsilon-1',  'stNo-456', 'p-0.5']
colorList = [ "#3498db", "#9b59b6", "k", "r", "orange", "#2ecc71"] # darkviolet
markerList = ['o', '+', 'h', '^', '*', 's', 'h', '2', 'p']
days = [i for i in range(30)]

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

    fig, (ax1, ax2) = plt.subplots(ncols=2
                                    , figsize=(18, 8)
                                   , sharex=True
                                   )

    #for index in range(len(methodlist)):

        # if methodlist[index] == 'epsilon-1':
        #     continue
        #
        # x, y = smooth_xy(days, data[index])  # 光滑处理
        #
        # ax.plot(x
        #         , y
        #         , linewidth=4
        #         , color=colorList[index]  # 线的颜色
        #         , marker=markerList[index]  # marker形状
        #         , ms=10  # marker size
        #         , mec=colorList[index]  # marker边框颜色
        #         , linestyle='-'
        #         #, mfc=colorList[index]  # marker实心颜色
        #         , mfc='white'  # marker实心颜色
        #         , label=methodlist[index]  # legend显示
        #         #, clip_on=False
        #         )

    '''箱线图'''

    data = np.array(data).swapaxes(0, 1)

    df = pd.DataFrame(data, columns=methodlist)
    df[df == -0.0] = 0



    sns.violinplot(data=df
                   , ax=ax1  # 指定子图
                   # , palette = my_pal  # 设置小提琴图颜色
                   , palette = colorList
                   )


    ax1.set_xlabel("Days", fontsize=32, fontproperties='Times New Roman'
                  #, fontweight='bold'
                  )
    ax1.set_ylabel('Inventory costs', fontsize=32, fontproperties='Times New Roman'
                  #, fontweight='bold'
                  #, rotation = 0
                  #, labelpad=15
                  )

    var = df.var().tolist()
    ax2.bar(range(len(methodlist)), var, 0.8, color=colorList)
    ax2.set_xlabel("Variance", fontsize=32, fontproperties='Times New Roman'
                  #, fontweight='bold'
                  )

    # 坐标上刻度字体格式设置
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # 指定x,y坐标轴上的刻度标签，并指定大小和x轴标签的旋转角度
    ax1.tick_params(axis="x", labelsize=22, labelrotation=15)
    ax1.tick_params(axis="y", labelsize=22)
    ax2.tick_params(axis="x", labelsize=22, labelrotation=15)
    ax2.tick_params(axis="y", labelsize=22)

    # ax1.patch.set_facecolor((0.925, 0.125, 0.90, 0.035))
    # ax2.patch.set_facecolor((0.925, 0.125, 0.90, 0.035))
    plt.subplots_adjust(wspace=0.30)
    # plt.xticks(fontproperties='Times New Roman', fontsize=20)
    # plt.yticks(fontproperties='Times New Roman', fontsize=20)
    ax1.grid(True, linestyle=':',alpha=0.6)
    ax2.grid(True, linestyle=':',alpha=0.6)

    plt.grid(True, linestyle=':', alpha=0.6)
    #plt.legend(prop={'size': 24, 'family': 'Times New Roman'}, loc='upper left', framealpha=0.6)
    fig.tight_layout()
    plt.savefig('./pic/inventory_costs.png'
                #, format='pdf'
                , bbox_inches='tight'
                , pad_inches=0.1, dpi=500)
    plt.show()



