import numpy as np
import matplotlib.pyplot as plt
day_num = 30
# RL，全部给库存，0.8给库存，完全greedy，剩余的456给库存，0.5的参与匹配
match_pro1 = 30
match_pro2 = 1.5
match_pro3 = 0.2
self_cost = -1.5
inv_cost = 2

colorList1 = [ "#3498db", "#9b59b6","k", "r", "orange", "#2ecc71"] # darkviolet
colorList2 = [ "#3498db", "#9b59b6","k", "r", "orange", "#2ecc71"] # darkviolet
markerList = ['o', '+', 'h', '^', '*', 's', 'h', '2', 'p']
methodlist = ['inv RL', 'epsilon-0.3','epsilon-0.7', 'epsilon-1',  'stNo-456', 'p-0.5']


def get_R1_1():
    file_name = "AllData.txt"
    data = list()
    with open(file_name, 'r') as f:
        for line in f:
            data_ = line.strip().split(', ')
            data.append([float(data_str) for data_str in data_])

    R1_1_data = data[0: 6]

    return R1_1_data

def get_R1_2():
    file_name = "AllData.txt"
    data = list()
    with open(file_name, 'r') as f:
        for line in f:
            data_ = line.strip().split(', ')
            data.append([float(data_str) for data_str in data_])

    R1_2_data = data[6: 12]

    return R1_2_data

def get_R2():
    file_name = "AllData.txt"
    data = list()
    with open(file_name, 'r') as f:
        for line in f:
            data_ = line.strip().split(', ')
            data.append([float(data_str) for data_str in data_])
    R2_data = data[12: 18]

    return  R2_data

def get_R3():
    file_name = "AllData.txt"
    data = list()
    with open(file_name, 'r') as f:
        for line in f:
            data_ = line.strip().split(', ')
            data.append([float(data_str) for data_str in data_])
    R3_data = data[18: 24]

    return  R3_data


def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c




if __name__ == "__main__":

    days = [i for i in range(1, 31, 1)]
    DRL_reward, DRL_cost = [], []
    R1_1_data = get_R1_1()
    R1_2_data = get_R1_2()
    R2_data = get_R2()
    R3_data = get_R3()


    for index in range(len(R1_1_data)):
        R1_1_data[index].reverse()
        DRL_cost = list_add(list_add(R1_2_data[index], R2_data[index]), R3_data[index])
        DRL_cost.reverse()

        DRL_reward = np.absolute(np.array(R1_1_data[index]))
        DRL_cost =np.absolute(np.array(DRL_cost))

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.bar(days, DRL_reward, color=colorList1[index], label=methodlist[index] + '_reward')
        plt.bar(days, -DRL_cost, color='gray', label=methodlist[index] + '_cost')

        ax.set_xlabel("Days", fontsize=32, fontproperties='Times New Roman'
                      # , fontweight='bold'
                      )
        ax.set_ylabel("Rewards and costs", fontsize=32, fontproperties='Times New Roman'
                      # , fontweight='bold'
                      )

        xticks = [1, 5, 10, 15, 20, 25, 30]
        plt.xticks(xticks,fontproperties='Times New Roman', fontsize=20)
        plt.yticks(fontproperties='Times New Roman', fontsize=20)

        plt.legend(prop={'size': 32, 'family': 'Times New Roman'})
        fig.tight_layout()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig('./pic/reward_cost_' + methodlist[index] + '.png'
                    # , format='pdf'
                    , bbox_inches='tight'
                    , pad_inches=0.1, dpi=500)


        plt.show()









