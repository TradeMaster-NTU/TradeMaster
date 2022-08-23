import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# here is a few function to draw the picutres
def evaualte(df):
    daily_return=df["daily_return"]
    neg_ret_lst=df[df["daily_return"]<0]["daily_return"]
    sharpe_ratio=np.mean(daily_return)/np.std(daily_return)*(len(df)**0.5)
    vol=np.std(daily_return)
    mdd=max((max(df["Cumlative Return"])-df["Cumlative Return"])/max(df["Cumlative Return"]))
    cr=np.sum(daily_return)/mdd
    sor=np.sum(daily_return)/np.std(neg_ret_lst)/np.sqrt(len(daily_return))
    df["sharpe_ratio"]=sharpe_ratio
    df["vol"]=vol
    df["mdd"]=mdd
    df["cr"]=cr
    df["sor"]=sor
    return df
def plot_pictures(models,path):
    algorithms=["A2C","PPO","SAC","SARL","DT","AlphaMix+"]
    tr=[]
    sr=[]
    for model in models:
        data=(model["Cumlative Return"]-1)*30+1
        tr.append(data)
        sr.append(model["sharpe_ratio"]*5-4)
    colors=['moccasin','aquamarine','#dbc2ec','orchid','lightskyblue','pink','orange']
    xticks=np.arange(len(algorithms))
    xticks = xticks *3
    fig, ax = plt.subplots(figsize=(7.45, 4))
    ax.yaxis.grid()
    ax.bar(xticks-0.5, tr, width=1, label="TR", color="lightskyblue")
    ax.bar(xticks+0.5, sr, width=1, label="TR", color="pink")
    plt.ylabel('Score',fontsize="xx-large")
    plt.xticks(xticks, algorithms,fontsize="xx-large")
    plt.axhline(1,label="average")# plt.xlim(-2, 16)
    plt.legend(["average",'TR', 'SR'], loc='upper center', ncol = 3,bbox_to_anchor=(-1, 1.2,3,0),fontsize="xx-large")
    plt.savefig(path,bbox_inches = 'tight' )
# notice that we do not provide the csv file so you need to switch the path into your own file path
A2C=pd.read_csv("./A2C average daily_return.csv",index_col=0)
SARL=pd.read_csv("./SARL average daily_return.csv",index_col=0)
DeepTrader=pd.read_csv("./DeepTrader average daily_return.csv",index_col=0)
PPO=pd.read_csv("./PPO average daily_return.csv",index_col=0)
SAC=pd.read_csv("./SAC average daily_return.csv",index_col=0)
AlphaMix=pd.read_csv("AlphaMix+.csv",index_col=0)
average=pd.read_csv("average.csv",index_col=0)

# here is one example of how the file should look like
"""
daily_return	date	reward
90	0.049582	2021-04-01 23:59:59	1.049582
91	0.017885	2021-04-02 23:59:59	1.017885
92	-0.018915	2021-04-03 23:59:59	0.981085
93	0.017644	2021-04-04 23:59:59	1.017644
94	0.151859	2021-04-05 23:59:59	1.151859
...	...	...	...
146	-0.042859	2021-05-27 23:59:59	0.957141
147	-0.051505	2021-05-28 23:59:59	0.948495
148	-0.049628	2021-05-29 23:59:59	0.950372
149	0.036778	2021-05-30 23:59:59	1.036778
150	0.078026	2021-05-31 23:59:59	1.078026
"""
new_models=[]
for model in [average,A2C,SARL,DeepTrader,PPO,SAC,AlphaMix]:
    new_model=evaualte(model)
    y_list=[1]
    for x in list(model.reward):
        y=y_list[-1]*x
        y_list.append(y)
    y_list=y_list[1:]
    new_model["Cumlative Return"]=y_list
    new_models.append(new_model)
average=average.iloc[-1]
A2C=(A2C.iloc[-1])/np.abs(average)
DDPG=(SARL.iloc[-1])/np.abs(average)
PG=(DeepTrader.iloc[-1])/np.abs(average)
PPO=(PPO.iloc[-1])/np.abs(average)
SAC=(SAC.iloc[-1])/np.abs(average)
RLmix=(AlphaMix.iloc[-1])/np.abs(average)
average=(average)/np.abs(average)
path=".exen.pdf"
plot_pictures(new_models,path)