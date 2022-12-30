import argparse
import sys

sys.path.append(".")
from env.AT.AT import TradingEnv
from agent.DeepScalper.model import Net
import numpy as np
import torch
from torch import nn
import yaml
import os
import pandas as pd
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_nodes",
                    type=int,
                    default=256,
                    help="the number of transcation we learn at a time")
parser.add_argument("--batch_size",
                    type=int,
                    default=32,
                    help="the number of transcation we learn at a time")
parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate")
parser.add_argument("--epsilon",
                    type=float,
                    default=0.9,
                    help="the learning rate")
parser.add_argument("--gamma",
                    type=float,
                    default=0.9,
                    help="the learning rate")
parser.add_argument(
    "--target_freq",
    type=int,
    default=50,
    help="the number of updates before the eval could be as same as the target"
)
parser.add_argument(
    "--memory_capacity",
    type=int,
    default=2000,
    help="the number of updates before the eval could be as same as the target"
)
parser.add_argument("--train_env_config",
                    type=str,
                    default="config/input_config/env/AT/DeepScalper/train.yml",
                    help="the dict for storing env config")
parser.add_argument("--valid_env_config",
                    type=str,
                    default="config/input_config/env/AT/DeepScalper/valid.yml",
                    help="the dict for storing env config")
parser.add_argument("--test_env_config",
                    type=str,
                    default="config/input_config/env/AT/DeepScalper/test.yml",
                    help="the dict for storing env config")
parser.add_argument("--test_style_env_config",
                    type=str,
                    default="config/input_config/env/AT/DeepScalper/test_style.yml",
                    help="the dict for storing env config")


parser.add_argument(
    "--future_loss_weights",
    type=float,
    default=0.2,
    help="the weights for future loss",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="result/AT/trained_model",
    help="the path for storing the trained model",
)
parser.add_argument(
    "--result_path",
    type=str,
    default="result/AT/test_result",
    help="the path for storing the test result",
)
parser.add_argument(
    "--test_style",
    type=int,
    default=-1,
    help="test agent with market data of a specific style: 0-bear 1-stag 2-bull",
)


def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value
def load_style_yaml(yaml_path,style):
    curPath = os.path.abspath('.')
    yaml_path = os.path.join(curPath, yaml_path)
    f = open(yaml_path, 'r', encoding='utf-8')
    cfg = f.read()
    d = yaml.load(cfg, Loader=yaml.FullLoader)
    data=pd.read_csv(d["df_path"])
    # data['index_by_tick']=data.index
    data=data.reset_index()
    data=data.loc[data['label'] == style, :]
    def get_styled_intervals_and_gives_new_index(data):
        index_by_tick_list=[]
        index_by_tick=[]
        date=data['date'].to_list()
        last_date=date[0]
        date_counter=0
        index = data['index'].to_list()
        last_value = index[0] - 1
        last_index = 0
        intervals = []
        for i in range(data.shape[0]):
            if last_value != index[i] - 1:
                date_counter = -1
                intervals.append([last_index, i])
                last_value = index[i]
                last_index = i
                index_by_tick_list.append(index_by_tick)
                index_by_tick=[]
            if date[i]!=last_date:
                date_counter+=1
            index_by_tick.append(date_counter)
            last_value = index[i]
            last_date = date[i]
        intervals.append([last_index, data.shape[0]])
        index_by_tick_list.append(index_by_tick)
        return intervals,index_by_tick_list
    intervals,index_by_tick_list=get_styled_intervals_and_gives_new_index(data)
    data.drop(columns=['index'])
    if not os.path.exists('temp'):
        os.makedirs('temp')
    d_list=[]
    for i,interval in enumerate(intervals):
        data_temp=data.iloc[interval[0]:interval[1],:]
        data_temp.index=index_by_tick_list[i]
        data_temp.to_csv('temp/'+str(style)+'_'+str(i)+'.csv')
        if max(index_by_tick_list[i])<d['backward_num_day']:
            print('This segment length is less tan the length_day in config so it won\'t be tested')
            continue
        temp_d=copy.deepcopy(d)
        temp_d["df_path"]='temp/'+str(style)+'_'+str(i)+'.csv'
        d_list.append(temp_d)
    return d_list

#foo.py

# do something with args


class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        self.model_path = args.model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.train_ev_instance = TradingEnv(
            read_yaml_to_dict(args.train_env_config))
        self.valid_ev_instance = TradingEnv(
            read_yaml_to_dict(args.valid_env_config))
        self.test_ev_instance = TradingEnv(
            read_yaml_to_dict(args.test_env_config))
        if args.test_style!=-1:
            self.test_style_env_configs = load_style_yaml(args.test_style_env_config,args.test_style)
            self.test_style_env_instances = [TradingEnv(config) for config in self.test_style_env_configs]
        self.n_action = self.train_ev_instance.action_space.n
        self.n_state = self.train_ev_instance.observation_space.shape[0]

        self.eval_net, self.target_net = Net(
            self.n_state, self.n_action, args.hidden_nodes), Net(
                self.n_state, self.n_action,
                args.hidden_nodes)  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory_capacity = args.memory_capacity
        self.memory = np.zeros(
            (args.memory_capacity,
             self.n_state * 2 + 3))  # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(),
            lr=args.lr)  # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.epsilon = args.epsilon
        self.target_freq = args.target_freq
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.future_loss_weights = args.future_loss_weights
        self.result_path = args.result_path

    def choose_action(self, x):  # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x),
                            0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform(
        ) < self.epsilon:  # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value, info = self.eval_net.forward(
                x)  # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(
                actions_value,
                1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]  # 输出action的第一个数
        else:  # 随机选择动作
            action = np.random.randint(
                0, self.n_action)  # 这里action随机等于0或1 (N_ACTIONS = 2)
        return action  # 返回选择的动作 (0或1)

    def choose_action_test(self, x):  # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x),
                            0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度

        actions_value, info = self.eval_net.forward(
            x)  # 通过对评估网络输入状态x，前向传播获得动作值
        action = torch.max(
            actions_value,
            1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
        action = action[0]  # 输出action的第一个数

        return action  # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_,
                         info):  # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r, info], s_))  # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % self.memory_capacity  # 获取transition要置入的行数
        self.memory[index, :] = transition  # 置入transition
        self.memory_counter += 1  # memory_counter自加1

    def learn(self):  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % self.target_freq == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(
                self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1  # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(
            self.memory_capacity, self.batch_size)  # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[
            sample_index, :]  # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :self.n_state])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, self.n_state:self.n_state +
                                        1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, self.n_state + 1:self.n_state + 2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_state:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列
        b_info = torch.FloatTensor(b_memory[:,
                                            self.n_state + 2:self.n_state + 3])
        # 将32个info抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行1列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新

        q_eval = self.eval_net(b_s)[0].gather(1, b_a)
        v_eval = self.eval_net(b_s)[1]
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_)[0].detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        loss_future = self.loss_func(v_eval, b_info)
        loss = loss + self.future_loss_weights * loss_future
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数

    def train_with_valid(self, num_epoch=2):
        valid_score_list = []
        for i in range(num_epoch):
            print('<<<<<<<<<Episode: %s' % i)
            s = self.train_ev_instance.reset()
            episode_reward_sum = 0
            while True:
                a = self.choose_action(s)
                s_, r, done, info = self.train_ev_instance.step(a)
                self.store_transition(s, a, r, s_, info["volidality"])
                episode_reward_sum += r
                s = s_
                if self.memory_counter > self.memory_capacity:
                    self.learn()
                if done:
                    print('episode%s---reward_sum: %s' %
                          (i, round(episode_reward_sum, 2)))
                    break
            all_model_path = self.model_path + "/all_model/"
            if not os.path.exists(all_model_path):
                os.makedirs(all_model_path)
            torch.save(self.eval_net,
                       all_model_path + "num_epoch_{}.pth".format(i))
            # valid
            s = self.valid_ev_instance.reset()
            episode_reward_sum = 0
            done = False
            while not done:
                a = self.choose_action_test(s)
                s_, r, done, info = self.valid_ev_instance.step(a)
                episode_reward_sum += r
                s = s_
            valid_score_list.append(episode_reward_sum)
        index = valid_score_list.index(np.max(valid_score_list))
        model_path = all_model_path + "num_epoch_{}.pth".format(index)
        self.eval_net = torch.load(model_path)
        best_model_path = self.model_path + "/best_model/"
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        torch.save(self.eval_net, best_model_path + "best_model.pth")

    def test(self):
        s = self.test_ev_instance.reset()
        done = False
        while not done:
            a = self.choose_action_test(s)
            s_, r, done, info = self.test_ev_instance.step(a)
            s = s_
        rewards = self.test_ev_instance.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_ev_instance.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        df.to_csv(self.result_path + "/result.csv")
        # print(self.test_ev_instance.compound_memory)

    def test_style(self,style):
        best_model_path = self.model_path + "/best_model/"
        model_path = best_model_path + "best_model.pth"
        self.eval_net = torch.load(model_path)
        print('running on ' + str(len(self.test_style_env_instances)) + ' data slices')
        for i in range(len(self.test_style_env_instances)):
            s =self.test_style_env_instances[i].reset()
            done = False
            while not done:
                a = self.choose_action_test(s)
                s_, r, done, info = self.test_style_env_instances[i].step(a)
                s = s_
            rewards = self.test_style_env_instances[i].save_asset_memory()
            assets = rewards["total assets"].values
            df_return = self.test_style_env_instances[i].save_portfolio_return_memory()
            daily_return = df_return.daily_return.values
            df = pd.DataFrame()
            df["daily_return"] = daily_return
            df["total assets"] = assets
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)
            df.to_csv(self.result_path + "/style_"+str(style)+'_result_'+str(i)+".csv")

if __name__ == "__main__":
    args = parser.parse_args()
    a = DQN(args)
    if args.test_style != -1:
        print('test for style ' + str(args.test_style))
        # a.test()
        a.test_style(args.test_style)
        # shutil.rmtree('temp')
    else:
        a.train_with_valid()
        a.test()