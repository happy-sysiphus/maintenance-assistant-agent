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
import random
import warnings
error_list = [
    # 'E012' ,
    # 'E016',
    # 'E028',
    'E029'
              ]
warnings.filterwarnings("ignore")
random.seed(0)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#matplotlib.rcParams['text.usetex'] = True
config = {
            "font.family": 'serif',
            #"font.size": 18,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
         }
rcParams.update(config)


method_list = [
    'BayesianRidge', 'GRU','LightGBM',  'LSTM', 'NBEATS','RandomForest','RNN','TCN', 'TFT'
    # , 'LinearRegression',

]
#method_list1 = ['Basic RNN', 'LSTM', 'GRU ','N-BEATS ', 'TCN','TFT']
colorList = ['r', 'k', 'y', 'g', 'b', 'purple', 'c', 'm', 'brown'
    # , 'orange'
             ]  # darkviolet
markerList = ['o', '^', 'h', '+', '*', 's', 'h', '2', 'p'
    # , '4'
              ]

if __name__ == "__main__":
    for index in range(len(error_list)):
        root = 'E:/01读博/小论文/Benchmark paper/实验/实验1/' + error_list[index]+'/ALL'
        i = 0
        acc_time_list = []
        for path, dir_list, file_list in os.walk(root):
            for file_name in file_list:
                file = os.path.join(path, file_name)
                df = pd.read_csv(file).iloc[:, [2]]
                acc_time_list.append(df.values.T[0].tolist())
                i = i + 1
                # print()

        fig, ax = plt.subplots()

        for index in range(len(method_list)):
            # ax.plot([*range(10)]+1
            #         , cpu_time_list[index]
            #         , linewidth=2.3
            #         , color=colorList[index]  # 线的颜色
            #         , marker=markerList[index]  # marker形状
            #         , ms=5  # marker size
            #         , mec=colorList[index]  # marker边框颜色
            #         , linestyle='-'
            #         , mfc=colorList[index]  # marker实心颜色
            #         , label=method_list[index]  # legend显示
            #         )

            ax.scatter(range(1, 20, 1)
                       , acc_time_list[index]
                       , s=150
                       # , linewidth=2.3
                       , color=colorList[index]  # 线的颜色
                       , marker=markerList[index]  # marker形状
                       # , ms=5  # marker size
                       # , mec=colorList[index]  # marker边框颜色
                       # , linestyle='-'
                       # , mfc=colorList[index]  # marker实心颜色
                       , label=method_list[index]  # legend显示
                       , alpha=0.5
                       )

        # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')  # 科学计数法
        ax.set_xlabel("Instance index", fontsize=24, fontproperties='Times New Roman', fontweight='bold')
        ax.set_ylabel('AME', fontsize=24, fontproperties='Times New Roman', fontweight='bold'
                      # , rotation=0
                      , labelpad=15
                      )

        plt.xticks(range(1, 20, 1), fontproperties='Times New Roman', fontsize=16)
        plt.yticks(fontproperties='Times New Roman', fontsize=16)
        plt.legend(prop={'size': 10, 'family': 'Times New Roman'}
                   , loc='center right'  # 'upper right'
                   , framealpha=0.5
                   )
        fig.tight_layout()
        plt.savefig('ACC_benchmark1.pdf'
                    , format="pdf"
                    , bbox_inches='tight'
                    , pad_inches=0.1, dpi=500)
        plt.show()


