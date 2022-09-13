from audioop import bias
from enum import auto
import sys
sys.path.append(".")
from agent.DeepScalper.model import Net
import numpy as np
import torch
from torch import nn
model_a = torch.load('/mnt/d/TradeMaster-1/result/AT/trained_model/best_model/best_model.pth').cpu()
model_a.eval()

torch.set_printoptions(threshold=np.inf)
# Display all model layer weights
# txt=""
# for name, para in model_a.named_parameters():
#     print('{}: {}'.format(name, para)+"\n")
#     print('{}: {}'.format(name, para.shape)+"\n")
#     txt+='{}: {}'.format(name, para)+"\n"
# f = open("/mnt/d/TradeMaster-1/result/AT/trained_model/parameter_printer/weights.txt",'a')
# f.write(txt)
# f.close()
paras=[]
names=[]
for name, para in model_a.named_parameters():
    para=para.reshape(-1)
    paras.append(para.detach().numpy().tolist())
    names.append(name)

# paras=torch.cat(paras,dim=1).reshape(-1)
# bias weights 放到一起 weights为偶数 bias为奇数


print(names)
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# colors = [plt.cm.Spectral(i/float(len(paras)-1)) for i in range(len(paras))]
# n, bins, patches = plt.hist(paras, 10, stacked=True, density=False, color=colors[:len(paras)])
# plt.legend({group:col for group, col in zip(names, colors[:len(paras)])},loc='lower center',ncol=int(len(paras)/2),fontsize=8,bbox_to_anchor=(0.25, 1., 0.5, 0.5))
# plt.savefig("/mnt/d/TradeMaster-1/result/AT/trained_model/parameter_printer/single_layer_distribution.pdf")
from matplotlib.ticker import FuncFormatter

linear_weights=paras[0:len(paras):2]
linear_name=names[0:len(paras):2]
bias_weights=paras[1:len(paras):2]
bias_name=names[1:len(paras):2]
colors = [plt.cm.Spectral(i/float(len(linear_weights)-1)) for i in range(len(linear_weights))]
colors=["pink",plt.cm.Spectral(1/float(len(linear_weights)-1)),"#c7fdb5","#cea2fd"]
# n, bins, patches=plt.hist(linear_weights,10, stacked=True, density=False, color=colors[:len(linear_weights)],edgecolor='black',cumulative=False)





n, bins, patches=plt.hist(bias_weights,10, stacked=True, density=True, color=colors[:len(bias_weights)],edgecolor='black',cumulative=False)
print(bins[-1]-bins[0])
def pencetail(weights,bins):
    weights_sum=0
    for single_layer_weights in weights:
        for weight in single_layer_weights:
            weights_sum=weights_sum+1
    percentages=[]
    for i in range(len(bins)-1):
        percentage=0
        for single_layer_weights in weights:
            for weight in single_layer_weights:
                if (weight>=bins[i] and weight <bins[i+1]):
                    percentage=percentage+1
        percentage=percentage/weights_sum
        percentages.append(percentage)
    return percentages


percentages=pencetail(bias_weights,bins)
for i in range(10):
    plt.text(bins[i]+(bins[1]-bins[0])/2,percentages[i]*38+0.01, str(round(percentages[i]*100,2)), ha='center', va= 'bottom')
def to_percent(y,position):
    return str(round(y*(bins[-1]-bins[0])*10,2))+"%"#这里可以用round（）函数设置取几位小数
fomatter=FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(fomatter)
plt.legend({group:col for group, col in zip(bias_name, colors[:len(bias_name)])},loc='lower center',
ncol=int(len(paras)/2),fontsize=8,bbox_to_anchor=(0.25, 1., 0.5, 0.5))
plt.ylabel("Percentage")
plt.xlabel("Value of bias")
plt.savefig("/mnt/d/TradeMaster-1/result/AT/trained_model/parameter_printer/single_layer_bias_distribution.pdf")


















# weights 图像
# n, bins, patches=plt.hist(linear_weights,10, stacked=True, density=True, color=colors[:len(linear_weights)],edgecolor='black',cumulative=False)
# def pencetail(weights,bins):
#     weights_sum=0
#     for single_layer_weights in weights:
#         for weight in single_layer_weights:
#             weights_sum=weights_sum+1
#     percentages=[]
#     for i in range(len(bins)-1):
#         percentage=0
#         for single_layer_weights in weights:
#             for weight in single_layer_weights:
#                 if (weight>=bins[i] and weight <bins[i+1]):
#                     percentage=percentage+1
#         percentage=percentage/weights_sum
#         percentages.append(percentage)
#     return percentages
# percentages=pencetail(linear_weights,bins)
# for i in range(10):
#     plt.text(bins[i]+(bins[1]-bins[0])/2,percentages[i]*10+0.1, round(percentages[i]*100,2), ha='center', va= 'bottom')
#     # plt.text(bins[i]+(bins[1]-bins[0])/2, n[i,:].sum()*1.01, int(n[i,:].sum()), ha='center', va= 'bottom')
# def to_percent(y,position):
#     return str(10*y)+"%"#这里可以用round（）函数设置取几位小数
# fomatter=FuncFormatter(to_percent)
# plt.gca().yaxis.set_major_formatter(fomatter)
# plt.legend({group:col for group, col in zip(linear_name, colors[:len(linear_name)])},loc='lower center',
# ncol=int(len(paras)/2),fontsize=8,bbox_to_anchor=(0.25, 1., 0.5, 0.5))
# plt.ylabel("Percentage")
# plt.xlabel("Value of weights")

# plt.savefig("/mnt/d/TradeMaster-1/result/AT/trained_model/parameter_printer/single_layer_weights_distribution.pdf")



































# kde graph 
# plt.figure(figsize=(16,10))
# sns.kdeplot(paras, shade=True, color="g", label="parameters", alpha=.7)
# plt.title('Density Plot of parameters in deepscalper', fontsize=22)
# plt.legend()
# plt.show()
# plt.savefig("/mnt/d/TradeMaster-1/result/AT/trained_model/parameter_printer/all_layer_distribution.pdf")