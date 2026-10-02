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


method_list = ['BayesianRidge',
               # 'LinearRegression ',
               'Random forest',
               'LightGBM ']
#method_list1 = ['Basic RNN', 'LSTM', 'GRU ','N-BEATS ', 'TCN','TFT']
colorList = ['r', 'k', 'y', 'g', 'b', 'purple', 'c', 'm', 'brown']  # darkviolet
markerList = ['o', '^', 'h', '+', '*', 's', 'h', '2', 'p']


cpu_time_list = [
        [0.8837,0.9061,2.3849,0.2762,0.2450,0.4028,0.2460,0.5448,0.8245,0.6092,0.7793,1.0222,0.9495,0.7820,0.6143 ],
        # [10.9591,22.4696,297.6509,85.1811,5495.2570,1.2711,737.7791,6.9958,2.5692,2.2833,91.9998,2.8561,2941.7444,3552.6655,5.2878],
        [0.3259,0.8867,2.3850,0.2440,0.2716,0.2925,0.2793,0.3601,0.6864,0.5441,0.9047,0.7534,2.7236,0.1340,0.6280,],
        [0.376598045,0.846998295,2.384920061,0.202501882,0.265007173,0.289020337,0.078941155,0.303689216,0.726637554,0.472862981,1.474372578,0.677010525,2.705489105,0.095315294,0.07221372],
]

# cpu_time_list1 = [
#     [404.56718428, 516.57187283, 629.95312474, 715.7968735, 829.15642595, 930.06093538, 1002.26389458, 1121.42968695, 1247.10156372, 1338.93119552],
#     [290.95781064, 364.92188041, 446.81093118, 515.08437595, 594.61250238, 672.60797899, 733.84687619, 821.9998296,  909.04682438, 980.48437324],
#     [286.0124964,  359.5421881, 439.74218736, 509.47812538, 585.74374962, 664.5015641, 727.14062622, 854.61636915, 903.80312998, 944.10704601],
#     [228.05780921, 289.0812505, 347.57656567, 397.13124919, 460.96406226, 516.54843843, 562.38749602, 628.26718986, 696.36874959, 754.04531286],
#     [248.9750005, 313.69374673, 324.95861516, 434.50625346, 500.53593419, 566.53750155, 620.4453145, 693.6828166, 772.8265554, 830.73281226]
# ]




if __name__ == "__main__":

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

        ax.scatter(range(1,16,1)
                , cpu_time_list[index]
                , s = 150
                #, linewidth=2.3
                , color=colorList[index]  # 线的颜色
                , marker=markerList[index]  # marker形状
                #, ms=5  # marker size
                #, mec=colorList[index]  # marker边框颜色
                #, linestyle='-'
                #, mfc=colorList[index]  # marker实心颜色
                , label=method_list[index]  # legend显示
                , alpha = 0.5
                )

    #ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')  # 科学计数法
    ax.set_xlabel("Instance index", fontsize=24, fontproperties='Times New Roman', fontweight='bold')
    ax.set_ylabel('AME', fontsize=24, fontproperties='Times New Roman', fontweight='bold'
                  #, rotation=0
                  , labelpad=15
                  )

    plt.xticks(range(1,16,1), fontproperties='Times New Roman', fontsize=16)
    plt.yticks(fontproperties='Times New Roman', fontsize=16)
    plt.legend(prop={'size': 16, 'family': 'Times New Roman'}
               , loc='best'   #'upper right'
               , framealpha=0.5
               )
    fig.tight_layout()
    plt.savefig('ACC_benchmark1.pdf'
                , format="pdf"
                , bbox_inches='tight'
                , pad_inches=0.1, dpi=500)
    plt.show()


