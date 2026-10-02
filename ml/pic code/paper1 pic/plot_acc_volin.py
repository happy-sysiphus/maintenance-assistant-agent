from matplotlib import rcParams
import random
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
sns.set(color_codes=True)
sns.set_style("whitegrid")
error_list = [
    'E01',
    'E02',
    'E03',
    # 'E04'
    ]
# warnings.filterwarnings("ignore")
# random.seed(0)
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# #matplotlib.rcParams['text.usetex'] = True
# config = {
#             "font.family": 'serif',
#             #"font.size": 18,
#             "mathtext.fontset": 'stix',
#             "font.serif": ['Times New Roman'],
#          }
# rcParams.update(config)


method_list= [
    'BayesianRidge', 'GRU','LightGBM', 'LSTM', 'NBEATS', 'RandomForest',
     'RNN', 'TCN', 'TFT'
              ]
colorList_ML = ['r', 'k', 'y', 'g', 'b', 'purple', 'c', 'm', 'brown']  # darkviolet
markerList = ['o', '^', 'h', '+', '*', 's', 'h', '2', 'p']

if __name__ == "__main__":
    for index in range(len(error_list)):
        root = 'E:/01读博/小论文/Benchmark paper/实验/实验1/' + error_list[index] + '/ALL'
        i = 0
        acc_list = []
        for path, dir_list, file_list in os.walk(root):
            for file_name in file_list:
                file = os.path.join(path, file_name)
                df = pd.read_csv(file).iloc[:, [3]]
                # df_t = df.values.T[0]
                # df_t.index = error_list[index]
                acc_list.append(df.values.T[0].tolist())
                i = i + 1
        acc_list_df = pd.DataFrame(acc_list, index=method_list)
        font1 = {'family': 'Times New Roman'
                 ,'weight': 'normal'
                 , 'size': 24
                 }
        # 生成子图
        fig, (ax) = plt.subplots(ncols=1
                                       # , figsize=(24, 8)
                                       # ,sharex=True
                                       )
        # 设置图片输出尺寸
        fig.set_size_inches(20, 6.5)  # 18 7
        # 小提琴图：颜色列表
        # method_list = ['GDE3', 'MODEA-OA', 'NSGA2','OW-MOSaDE', 'DQN&DDPG']
        # my_pal = {"GDE3": "r", "MODEA-OA": "K", "NSGA2": "y", "OW-MOSaDE": "g", "AMODE-DRL": "b"}
        sns.violinplot(data=acc_list_df.T,
                       # , ax=ax # 指定子图
                       # , palette = my_pal  # 设置小提琴图颜色
                       palette="Set3",
                       )
        # 指定纵坐标标签，并同时指定标签字体格式和字体大小
        ax.set_ylabel('mape', font1, fontweight='bold')



        # 坐标上刻度字体格式设置
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # 指定x,y坐标轴上的刻度标签，并指定大小和x轴标签的旋转角度
        ax.tick_params(axis="x", labelsize=15, labelrotation=0)
        ax.tick_params(axis="y", labelsize=15)

        # 指定两个子图之间的间距
        # plt.subplots_adjust(wspace=0.15)

        #plt.xlabel("Problem Index="+ str(i+1),font1,fontweight='bold')
        fig.text(0.5, 0.001, "Error " + error_list[index], ha='center', fontweight='bold', fontsize=24, fontproperties='Times New Roman')
        # fig.tight_layout()

        plt.savefig('ML+NN' + error_list[index] + '.png'
                    , format='png'
                    , bbox_inches='tight'
                    , pad_inches=0.1
                    , dpi=500)
        plt.show()

