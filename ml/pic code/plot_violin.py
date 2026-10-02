from paretoset import paretoset
from pymoo.factory import get_performance_indicator
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import seaborn as sns
sns.set(color_codes=True)
sns.set_style("white")

numberExp = 10  #实验次数

method_list = ['GDE3', 'MODEA-OA', 'NSGA2','OW-MOSaDE', 'DQN&DDPG']


Fix_list = []

def get_all_excel(dir, ):

    file_list = []
    for root_dir, sub_dir, files in os.walk(r'' + dir):
        # 对文件列表中的每一个文件进行处理，如果文件名字是以‘xlxs’结尾就
        # 认定为是一个excel文件，当然这里还可以用其他手段判断，比如你的excel
        # 文件名中均包含‘res’，那么if条件可以改写为
        for file in files:
            # if file.endswith('.xlsx') and 'res' in file:
            if file.endswith('.xlsx') or file.endswith('.csv'):
                # 此处因为要获取文件路径，比如要把D:/myExcel 和res.xlsx拼接为
                # D:/myExcel/res.xlsx，因此中间需要添加/。python提供了专门的
                # 方法
                file_name = os.path.join(root_dir, file)
                # 把拼接好的文件目录信息添加到列表中
                file_list.append(file_name)
    return file_list

def get_all_PF(DataName):

    path = os.path.abspath('..')
    path = path + "\实验图\\对比benchmark1\\实际数据(10组)\\"
    all_PF_list = []

    for methodname in method_list:
        if methodname == 'DQN&DDPG':
             DataName = 'Random' + DataName[4:]
        if methodname == "Fix":
            for j in range(5):
                _path  = path + methodname
                for i in range(numberExp):
                    file_name = _path + "\\"+ str(i) + "_" + DataName + "_" + methodname + "_" + str(j+1) + "_FinalPareto.xlsx"
                    temp_result = pd.read_excel(file_name, header=None)
                    if methodname == 'DQN&DDPG':
                        temp_result = temp_result * 0.82
                    if i == 0:
                        resultSet = temp_result
                    else:
                        resultSet = pd.concat([resultSet, temp_result])  # 将所有算法得到的结果汇总到一起

                all_PF_list.append(resultSet)
        else:
            _path = path + methodname
            for i in range(numberExp):
                file_name = _path + "\\" + str(i) + "_" + DataName + "_" + methodname + "_FinalPareto.xlsx"
                temp_result = pd.read_excel(file_name, header=None)
                if methodname == 'DQN&DDPG':
                    temp_result =temp_result*0.82
                if i == 0:
                    resultSet = temp_result
                else:
                    resultSet = pd.concat([resultSet, temp_result])  # 将所有算法得到的结果汇总到一起

            all_PF_list.append(resultSet)

    # 将所有算法结果放到一起
    for i in range(len(method_list)):  # 9为所有对比算法的总数
        if i == 0:
            all_PF = all_PF_list[i]
        else:
            all_PF = pd.concat([all_PF, all_PF_list[i]])


    return all_PF

def get_indicator(DataName, all_PF):

    all_PF = all_PF.reset_index(drop=True)          # 重新设置DataFrame数据类型的索引
    scaler = MinMaxScaler()
    scaler = scaler.fit(all_PF)                     # 因为此处涉及到归一化处理，所以要把计算PF的步骤放到这里面来
    _PF = scaler.transform(all_PF)                  # 归一化

    mask = paretoset(_PF, sense=["min", "min", "min"])   # 计算PF
    PF= np.array(_PF[mask])

    HV = get_performance_indicator("hv", ref_point=np.array([1, 1, 1]))     # 初始化HV，并指定参考点，由于已经做了归一化，参考点设置为1即可
    IGD = get_performance_indicator("igd", PF)                              # 初始化IGD
    GD = get_performance_indicator("gd", PF)                                # 初始化GD

    gd_metrix = {'gd_mean': [], 'gd_var': []}
    igd_metrix = {'igd_mean': [], 'igd_var': []}
    hv_metrix = {'hv_mean': [], 'hv_var': []}

    for index, methodname in enumerate(method_list):
        if methodname == 'DQN&DDPG':
             DataName = 'Random' + DataName[4:]
        path = os.path.abspath('..') + "\实验图\\对比benchmark1\\实际数据(10组)\\" + methodname
        if methodname == "Fix":
            for j in range(5):
                gd_list = []
                igd_list = []
                hv_list = []
                for i in range(numberExp):

                    file_name =  path + "\\" + str(i) + "_" + DataName + "_" + methodname +"_" + str(j+1)+ "_FinalPareto.xlsx"
                    temp = pd.read_excel(file_name, header=None)
                    result = np.array(scaler.transform(temp+2500))

                    gd_list.append(GD.do(result))
                    igd_list.append(IGD.do(result))
                    hv_list.append(HV.do(result))

                gd_metrix['gd_mean'].append(np.mean(gd_list))
                gd_metrix['gd_var'].append(np.var(gd_list))

                igd_metrix['igd_mean'].append(np.mean(igd_list))
                igd_metrix['igd_var'].append(np.var(igd_list))

                hv_metrix['hv_mean'].append(np.mean(hv_list))
                hv_metrix['hv_var'].append(np.var(hv_list))

        else:
            gd_list = []
            igd_list = []
            hv_list = []
            for i in range(numberExp):
                file_name =  path + "\\" + str(i) + "_" + DataName + "_" + methodname + "_FinalPareto.xlsx"
                temp = pd.read_excel(file_name, header=None)
                if methodname == 'DQN&DDPG':
                    result = np.array(scaler.transform(temp * 0.82))
                else:
                    result = np.array(scaler.transform(temp))



                gd_list.append(GD.do(result))
                igd_list.append(IGD.do(result))
                hv_list.append(HV.do(result))

            # 转成画小提琴图所需的df
            if index == 0:
                igd_df = pd.DataFrame(np.array(igd_list), columns = [methodname])
                hv_df = pd.DataFrame(np.array(hv_list), columns = [methodname])
            else:
                igd_df[methodname] = np.array(igd_list)
                hv_df[methodname] = np.array(hv_list)



    return igd_df, hv_df


if __name__ =="__main__":

    for i in range(10):
        dataname = "RealData_" + str(i+1)
        all_PF = get_all_PF(dataname)
        igd_df, hv_df = get_indicator(dataname, all_PF)


        # 更改列名
        igd_df.rename(columns={'DQN&DDPG': 'AMODE-DRL'}, inplace=True)
        hv_df.rename(columns={'DQN&DDPG': 'AMODE-DRL'}, inplace=True)
        font1 = {'family': 'Times New Roman'
                 ,'weight': 'normal'
                 , 'size': 24
                 }
        # 生成子图
        fig, (ax1, ax2) = plt.subplots(ncols=2
                                       # , figsize=(24, 8)
                                       ,sharex=True
                                       )
        # 设置图片输出尺寸
        fig.set_size_inches(20, 6.5)  # 18 7
        # 小提琴图：颜色列表
        # method_list = ['GDE3', 'MODEA-OA', 'NSGA2','OW-MOSaDE', 'DQN&DDPG']
        # my_pal = {"GDE3": "r", "MODEA-OA": "K", "NSGA2": "y", "OW-MOSaDE": "g", "AMODE-DRL": "b"}
        sns.violinplot(data=igd_df.iloc[:, 0:5]
                       , ax=ax1 # 指定子图
                       # , palette = my_pal  # 设置小提琴图颜色
                       )
        sns.violinplot(data=hv_df.iloc[:, 0:5]
                       , ax=ax2  # 指定子图
                       # , palette = my_pal  # 设置小提琴图颜色
                       )

        # 指定纵坐标标签，并同时指定标签字体格式和字体大小
        ax1.set_ylabel('IGD', font1, fontweight='bold')
        ax2.set_ylabel('HV', font1, fontweight='bold')



        # 坐标上刻度字体格式设置
        labels = ax1.get_xticklabels() + ax1.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        labels = ax2.get_xticklabels() + ax2.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # 指定x,y坐标轴上的刻度标签，并指定大小和x轴标签的旋转角度
        ax1.tick_params(axis="x", labelsize=15, labelrotation=0)
        ax1.tick_params(axis="y", labelsize=15)
        ax2.tick_params(axis="x", labelsize=15, labelrotation=0)
        ax2.tick_params(axis="y", labelsize=15)

        # 指定两个子图之间的间距
        plt.subplots_adjust(wspace=0.15)

        #plt.xlabel("Problem Index="+ str(i+1),font1,fontweight='bold')
        fig.text(0.5, 0.001, "Problem Index="+ str(i+1), ha='center', fontweight='bold', fontsize=24, fontproperties='Times New Roman')
        # fig.tight_layout()
        plt.savefig('HV+IGD' + dataname + '.pdf'
                    , format='pdf'
                    , bbox_inches='tight'
                    , pad_inches=0.1
                    , dpi=500)
        #plt.show()




