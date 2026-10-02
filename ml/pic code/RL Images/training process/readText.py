# 读txt，并生成图像
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

sns.set_style("white")

current_palette = sns.color_palette('deep')


file_list = ['reward01.txt', 'reward02.txt', 'reward03.txt']
data_list = list()
for file_name in file_list:
    with open(file_name, 'r') as f:
        data = list()
        for line in f:
            if line != "save parameters\n":
                line = line.strip().split('=')[-1].strip()
                # print("line: [" + line + "]")
                line = float(line)
                data.append(line)
        data_list.append(data)

for i in range(len(data_list)):
    #data_list[i] = [data_list[i][j] for j in range(len(data_list[i])) if j % 5 == 0][:600]
    weight = 0.8
    data_new = list()
    data_list[i] = data_list[i][:6000]
    last_data = data_list[i][0]
    for current_data in data_list[i]:
        temp_val = weight * last_data + (1 - weight) * current_data
        data_new.append(temp_val)
        last_data = temp_val
    data_list[i] =  [data * (1/weight) for data in data_new]

data_ = np.concatenate((data_list[0], data_list[1], data_list[2]))
episode = np.concatenate((range(len(data_list[0])), range(len(data_list[1])), range(len(data_list[2]))))

fig, ax = plt.subplots(figsize=(14,8))

sns.lineplot(x=episode, y=data_, color="#3498db", ax=ax)

# ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')  # 科学计数法
ax.set_xlabel("Episode", fontsize=32, fontproperties='Times New Roman'
              # , fontweight='bold'
              )
ax.set_ylabel('Average rewards', fontsize=32, fontproperties='Times New Roman'
              # , fontweight='bold'
              # , rotation = 0
              # , labelpad=15
              )

#ax.patch.set_facecolor((0.925, 0.125, 0.90, 0.035))
plt.xticks(fontproperties='Times New Roman', fontsize=20)
plt.yticks(fontproperties='Times New Roman', fontsize=20)
plt.grid(True, linestyle=':',alpha=0.6)
fig.tight_layout()
plt.savefig('../pic/training_process0.8.png'
            # , format='pdf'
            , bbox_inches='tight'
            , pad_inches=0.1, dpi=500)
plt.show()