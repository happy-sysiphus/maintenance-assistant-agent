import os
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import mpl_toolkits.mplot3d.axis3d as axis3d


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['text.usetex'] = True
config = {
            "font.family": 'serif',
            #"font.size": 18,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
         }
rcParams.update(config)


method_list = ['DDPG', 'DQN', 'DQN&DDPG', 'Fix', 'Random']
method_plot_list = ['DDPG', 'DQN', 'DQN&DDPG', 'Fix_1', 'Fix_2', 'Fix_3', 'Fix_4', 'Fix_5', 'Random']
colorList = ['red', 'k', 'b', 'limegreen', 'purple', 'y', 'c', 'm', 'blueviolet']  # blueviolet
markerList = ['o', '^', '*', '+', '*', 's', 'h', '2', 'p']
numberExp = 10  #实验次数


if __name__ == "__main__":

    for j in range(10):

        all_method_obj_1_mean_list, all_method_obj_1_min_list, all_method_obj_1_max_list, \
        all_method_obj_2_mean_list, all_method_obj_2_min_list, all_method_obj_2_max_list, \
        all_method_obj_3_mean_list, all_method_obj_3_min_list, all_method_obj_3_max_list, = get_obj("RandomData_" + str(j+1))

        #print(all_method_obj_3_mean_list)
        fig, ax = plt.subplots()
        for index in range(9):
            if index in [4, 5, 6, 7]:
                continue
            method = method_plot_list[index]
            if method == "DQN&DDPG":
                method = "DQN-DDPG"
            elif method in ["Fix_1", "Fix_2", "Fix_3", "Fix_4", "Fix_5"]:
                method = method.replace("_", "")

            ax.plot(range(100)
                    , all_method_obj_2_mean_list[index]
                    , linewidth = 2.3
                    , color= colorList[index]   # 线的颜色
                    , marker=markerList[index]  # marker形状
                    , ms=5  # marker size
                    , mec=colorList[index]  #marker边框颜色
                    , linestyle='-'
                    , mfc=colorList[index]  # marker实心颜色
                    , label = method        # legend显示
                    )

            ax.fill_between(range(100)
                            , all_method_obj_2_max_list[index]
                            , all_method_obj_2_min_list[index]
                            , color=colorList[index]
                            , alpha=0.1
                            )

        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')   # 科学计数法
        ax.set_xlabel("Iteration", fontsize=18, fontproperties='Times New Roman', fontweight='bold')
        ax.set_ylabel(r'$f_{3}$', fontsize=20, fontproperties='Times New Roman', fontweight='bold', rotation = 0, labelpad=15)
        plt.xticks(fontproperties='Times New Roman', fontsize=16)
        plt.yticks(fontproperties='Times New Roman', fontsize=16)
        plt.legend(prop={'size': 16, 'family': 'Times New Roman'}, loc = 'upper right', framealpha = 0.6)
        fig.tight_layout()
        plt.savefig('PCA.tiff'
                    , bbox_inches = 'tight'
                    , pad_inches=0, dpi=500)
        plt.show()





