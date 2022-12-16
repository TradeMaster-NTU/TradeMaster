from turtle import forward
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
# feature为通道数 长为时间 宽为tic
import numpy as np


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs,
                      n_outputs,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs,
                      n_outputs,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation))
        self.chomp2 = Chomp1d(padding)  #  裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1, self.conv2, self.chomp2,
                                 self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        self.bn = nn.BatchNorm1d(n_outputs)

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """

        out = self.net(x)
        # out = self.bn(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[
                i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [
                TemporalBlock(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size,
                              dropout=dropout),
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。
        
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


class SA(nn.Module):
    def __init__(self, C, N, K_l) -> None:
        # C should be the number of features output from the TCN,N should be the batch size, here indicates the number of tickers and
        # K_l represents the length
        super(SA, self).__init__()
        self.C = C
        self.N = N
        self.K_l = K_l
        self.W_1 = nn.Parameter(torch.randn(K_l))
        self.W_2 = nn.Parameter(torch.randn(C, K_l))
        self.W_3 = nn.Parameter(torch.randn(C))
        self.V_s = nn.Parameter(torch.randn(N, N))
        self.b_s = nn.Parameter(torch.randn(N, N))

    def forward(self, x):
        S_l = torch.matmul(torch.matmul(torch.matmul(x, self.W_1).T, self.W_2),
                           torch.matmul(self.W_3, x.transpose(0, 1)).T)
        return S_l


class GCN(nn.Module):
    # this module is designed only for the graph convolution part
    # It is a little trick here because we are using the adjclose as the metric to construct the convoluition
    def __init__(self, K_l) -> None:
        super(GCN, self).__init__()
        self.K_l = K_l
        self.theta = nn.Parameter(torch.randn(K_l, K_l))

    def forward(self, A, H_l):
        sum = np.sum(np.abs(A), axis=1)
        A = (A.T / sum).T
        A = torch.from_numpy(A).to(torch.float32)
        Z_l = torch.matmul(torch.matmul(A, H_l), self.theta)
        return Z_l


class IN(nn.Module):
    def __init__(self, N, num_features) -> None:
        super(IN, self).__init__()
        self.N = N
        self.linear = nn.Linear(num_features, 1)
        self.bn1 = nn.BatchNorm1d(num_features=num_features)

    def forward(self, S_l, Z_l, H_l):
        x = torch.matmul(S_l, Z_l)
        x = x + H_l
        x = x.reshape(self.N, -1)
        x = self.bn1(x)
        x = self.linear(x).squeeze()
        return x


class IN_value(nn.Module):
    def __init__(self, N, num_features) -> None:
        super(IN_value, self).__init__()
        self.N = N
        self.linear = nn.Linear(num_features, 1)
        self.linear2 = nn.Linear(2 * N, 1)

    def forward(self, S_l, Z_l, H_l, action):
        action = action.reshape(1, -1)
        x = torch.matmul(S_l, Z_l)
        x = x + H_l
        x = x.reshape(self.N, -1)
        x = torch.sigmoid(self.linear(x)).squeeze().unsqueeze(0)
        x = torch.cat((x, action), dim=1)
        x = self.linear2(x)
        return x


class asset_scoring(nn.Module):
    def __init__(self,
                 N,
                 K_l,
                 num_inputs,
                 num_channels,
                 kernel_size=2,
                 dropout=0.2) -> None:
        super(asset_scoring, self).__init__()
        self.TCN = TemporalConvNet(num_inputs, num_channels)
        self.SA = SA(num_channels[-1], N, K_l)
        self.GCN = GCN(K_l)
        self.IN = IN(N, num_channels[-1] * K_l)

    def forward(self, x, A):
        H_L = self.TCN(x)
        S_L = self.SA(H_L.transpose(0, 1))
        Z_L = self.GCN(A, H_L.transpose(0, 1))
        result = self.IN(S_L, Z_L, H_L.transpose(0, 1))
        return result


class asset_scoring_value(nn.Module):
    def __init__(self,
                 N,
                 K_l,
                 num_inputs,
                 num_channels,
                 kernel_size=2,
                 dropout=0.2) -> None:
        super(asset_scoring_value, self).__init__()
        self.TCN = TemporalConvNet(num_inputs, num_channels)
        self.SA = SA(num_channels[-1], N, K_l)
        self.GCN = GCN(K_l)
        self.IN_value = IN_value(N, num_channels[-1] * K_l)

    def forward(self, x, A, action):
        H_L = self.TCN(x)
        S_L = self.SA(H_L.transpose(0, 1))
        Z_L = self.GCN(A, H_L.transpose(0, 1))
        result = self.IN_value(S_L, Z_L, H_L.transpose(0, 1), action)
        return result


class market_scoring(nn.Module):
    def __init__(self, n_features, win=10, hidden_size=12) -> None:
        # super(market_scoring, self).__init__()
        # self.lstm = nn.LSTM(input_size=n_features,
        #                     hidden_size=hidden_size,
        #                     num_layers=1,
        #                     batch_first=True)
        # self.U1 = nn.Parameter(torch.randn(hidden_size, hidden_size * 2))
        # self.U2 = nn.Parameter(torch.randn(hidden_size, n_features))
        # self.V = nn.Parameter(torch.randn(hidden_size))
        # self.linear = nn.Linear(hidden_size, 2)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        super(market_scoring, self).__init__()
        self.in_features = n_features
        self.window_len = win
        self.hidden_dim = hidden_size

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size)
        self.attn1 = nn.Linear(2 * hidden_size, hidden_size)
        self.attn2 = nn.Linear(hidden_size, 1)

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # lstm_out, _ = self.lstm(x)

        # #lstm_out 为batch,length, feature dj30为(1,10,12)
        # H_K = lstm_out[:, -1, :]
        # eks = []
        # for k in range(lstm_out.shape[1]):
        #     h_k = lstm_out[:, k, :]
        #     h_kh_K = torch.cat((h_k, H_K), 1).reshape(-1, 1)
        #     multiplier = torch.matmul(self.U1, h_kh_K) + torch.matmul(
        #         self.U2, x[:, k, :].reshape(-1, 1))
        #     e_k = torch.matmul(self.V.reshape(1, -1), multiplier)
        #     eks.append(e_k)
        # eks = torch.cat(eks).unsqueeze(0)
        # print(eks.shape)
        # alpha_ks = nn.Softmax(dim=1)(eks)
        # H_K_bar = torch.matmul(alpha_ks.squeeze(2),
        #                        lstm_out[0, :, :]).squeeze()
        # print(H_K_bar.shape)
        # result = torch.sigmoid(self.linear((H_K_bar)).squeeze())
        X = x.permute(1, 0, 2)

        outputs, (h_n, c_n) = self.lstm(X)  # lstm version
        H_n = h_n.repeat((self.window_len, 1, 1))
        scores = self.attn2(
            torch.tanh(self.attn1(torch.cat([outputs, H_n],
                                            dim=2))))  # [L, B*N, 1]
        scores = scores.squeeze(2).transpose(1, 0)  # [B*N, L]
        attn_weights = torch.softmax(scores, dim=1)
        outputs = outputs.permute(1, 0, 2)  # [B*N, L, H]
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        embed = torch.relu(self.bn1(self.linear1(attn_embed).repeat(2, 1)))
        parameters = self.linear2(embed)
        parameters = parameters[0]
        # return parameters[:, 0], parameters[:, 1]   # mu, sigma
        return parameters.squeeze(-1)

        # return result


if __name__ == "__main__":
    # input = torch.randn(1, 10, 12)
    # net = market_scoring(12)
    # print(net(input).shape)
    input = torch.randn(29, 12, 10)
    A = np.random.randint(1, 2, size=(29, 29))
    net_new = asset_scoring(29, 10, 12, [12, 12, 12])
    print(net_new(input, A).shape)
