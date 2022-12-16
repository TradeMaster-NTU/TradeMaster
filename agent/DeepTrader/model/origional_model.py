import math

import torch
import torch.nn as nn


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in,
                                   c_out,
                                   kernel_size=(1, 1),
                                   padding=(0, 0),
                                   stride=(1, 1),
                                   bias=True)

    def forward(self, x):
        return self.mlp(x)


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super(GraphConvNet, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = nn.functional.dropout(h, self.dropout, training=self.training)
        return h


class SpatialAttentionLayer(nn.Module):
    def __init__(self, num_nodes, in_features, in_len):
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = nn.Linear(in_len, 1, bias=False)
        self.W2 = nn.Linear(in_features, in_len, bias=False)
        self.W3 = nn.Linear(in_features, 1, bias=False)
        self.V = nn.Linear(num_nodes, num_nodes)

        self.bn_w1 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w3 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w2 = nn.BatchNorm1d(num_features=num_nodes)

    def forward(self, inputs):
        # inputs: (batch, num_features, num_nodes, window_len)
        part1 = inputs.permute(0, 2, 1, 3)
        part2 = inputs.permute(0, 2, 3, 1)
        part1 = self.bn_w1(self.W1(part1).squeeze(-1))
        part1 = self.bn_w2(self.W2(part1))
        part2 = self.bn_w3(self.W3(part2).squeeze(-1)).permute(0, 2, 1)  #
        S = torch.softmax(self.V(torch.relu(torch.bmm(part1, part2))), dim=-1)
        return S


class SAGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_features,
                 hidden_dim,
                 window_len,
                 dropout=0.3,
                 kernel_size=2,
                 layers=4,
                 supports=None,
                 spatial_bool=True,
                 addaptiveadj=True,
                 aptinit=None):

        super(SAGCN, self).__init__()
        self.dropout = dropout
        self.layers = layers
        if spatial_bool:
            self.gcn_bool = True
            self.spatialattn_bool = True
        else:
            self.gcn_bool = False
            self.spatialattn_bool = False
        self.addaptiveadj = addaptiveadj

        self.tcns = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.sans = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.supports = supports

        self.start_conv = nn.Conv1d(in_features,
                                    hidden_dim,
                                    kernel_size=(1, 1))

        self.bn_start = nn.BatchNorm2d(hidden_dim)

        receptive_field = 1
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if self.gcn_bool and addaptiveadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec = nn.Parameter(torch.randn(num_nodes, 1),
                                            requires_grad=True)
                self.supports_len += 1

            else:
                raise NotImplementedError

        additional_scope = kernel_size - 1
        a_s_records = []
        dilation = 1
        for l in range(layers):
            tcn_sequence = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dim,
                          out_channels=hidden_dim,
                          kernel_size=(1, kernel_size),
                          dilation=dilation), nn.ReLU(), nn.Dropout(dropout),
                nn.BatchNorm2d(hidden_dim))

            self.tcns.append(tcn_sequence)

            self.residual_convs.append(
                nn.Conv1d(in_channels=hidden_dim,
                          out_channels=hidden_dim,
                          kernel_size=(1, 1)))

            self.bns.append(nn.BatchNorm2d(hidden_dim))

            if self.gcn_bool:
                self.gcns.append(
                    GraphConvNet(hidden_dim,
                                 hidden_dim,
                                 dropout,
                                 support_len=self.supports_len))

            dilation *= 2
            a_s_records.append(additional_scope)
            receptive_field += additional_scope
            additional_scope *= 2

        self.receptive_field = receptive_field
        if self.spatialattn_bool:
            for i in range(layers):
                self.sans.append(
                    SpatialAttentionLayer(num_nodes, hidden_dim,
                                          receptive_field - a_s_records[i]))
                receptive_field -= a_s_records[i]

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)  # [batch, feature, stocks, length]
        in_len = X.shape[3]
        if in_len < self.receptive_field:
            x = nn.functional.pad(X, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = X
        assert not torch.isnan(x).any()

        x = self.bn_start(self.start_conv(x))
        new_supports = None
        if self.gcn_bool and self.addaptiveadj and self.supports is not None:
            adp_matrix = torch.softmax(torch.relu(
                torch.mm(self.nodevec, self.nodevec.t())),
                                       dim=0)
            new_supports = self.supports + [adp_matrix]

        for i in range(self.layers):
            residual = self.residual_convs[i](x)
            x = self.tcns[i](x)
            if self.gcn_bool and self.supports is not None:
                if self.addaptiveadj:
                    x = self.gcns[i](x, new_supports)
                else:
                    x = self.gcns[i](x, self.supports)

            if self.spatialattn_bool:
                attn_weights = self.sans[i](x)
                x = torch.einsum('bnm, bfml->bfnl', (attn_weights, x))

            x = x + residual[:, :, :, -x.shape[3]:]

            x = self.bns[i](x)

        # (batch, num_nodes, hidden_dim)
        return x.squeeze(-1).permute(0, 2, 1)


class LiteTCN(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_size,
                 num_layers,
                 kernel_size=2,
                 dropout=0.4):
        super(LiteTCN, self).__init__()
        self.num_layers = num_layers
        self.tcns = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.start_conv = nn.Conv1d(in_features, hidden_size, kernel_size=1)
        self.end_conv = nn.Conv1d(hidden_size, 1, kernel_size=1)

        receptive_field = 1
        additional_scope = kernel_size - 1
        dilation = 1
        for l in range(num_layers):
            tcn_sequence = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size,
                          out_channels=hidden_size,
                          kernel_size=kernel_size,
                          dilation=dilation),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            self.tcns.append(tcn_sequence)

            self.bns.append(nn.BatchNorm1d(hidden_size))

            dilation *= 2
            receptive_field += additional_scope
            additional_scope *= 2
        self.receptive_field = receptive_field

    def forward(self, X):
        X = X.permute(0, 2, 1)
        in_len = X.shape[2]
        if in_len < self.receptive_field:
            x = nn.functional.pad(X, (self.receptive_field - in_len, 0))
        else:
            x = X

        x = self.start_conv(x)

        for i in range(self.num_layers):
            residual = x
            assert not torch.isnan(x).any()
            x = self.tcns[i](x)
            assert not torch.isnan(x).any()
            x = x + residual[:, :, -x.shape[-1]:]

            x = self.bns[i](x)
        assert not torch.isnan(x).any()
        x = self.end_conv(x)

        return torch.sigmoid(x.squeeze())


class ASU(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_features,
                 hidden_dim,
                 window_len,
                 dropout=0.3,
                 kernel_size=2,
                 layers=4,
                 supports=None,
                 spatial_bool=True,
                 addaptiveadj=True,
                 aptinit=None):
        super(ASU, self).__init__()
        self.sagcn = SAGCN(num_nodes, in_features, hidden_dim, window_len,
                           dropout, kernel_size, layers, supports,
                           spatial_bool, addaptiveadj, aptinit)
        self.linear1 = nn.Linear(hidden_dim, 1)

        self.bn1 = nn.BatchNorm1d(num_features=num_nodes)
        self.in1 = nn.InstanceNorm1d(num_features=num_nodes)

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
        )
        self.hidden_dim = hidden_dim

    def forward(self, inputs):
        """
        inputs: [batch, num_stock, window_len, num_features]
        mask: [batch, num_stock]
        outputs: [batch, scores]
        """
        x = self.sagcn(inputs)
        print(x.shape)
        x = self.bn1(x)
        x = self.linear1(x).squeeze(-1)
        score = 1 / ((-x).exp() + 1)
        return score


import torch
import torch.nn as nn


class MSU(nn.Module):
    def __init__(self, in_features, window_len, hidden_dim):
        super(MSU, self).__init__()
        self.in_features = in_features
        self.window_len = window_len
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim)
        self.attn1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, 1)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)

    def forward(self, X):
        """
        :X: [batch_size(B), window_len(L), in_features(I)]
        :return: Parameters: [batch, 2]
        """
        X = X.permute(1, 0, 2)

        outputs, (h_n, c_n) = self.lstm(X)  # lstm version
        H_n = h_n.repeat((self.window_len, 1, 1))
        scores = self.attn2(
            torch.tanh(self.attn1(torch.cat([outputs, H_n],
                                            dim=2))))  # [L, B*N, 1]
        scores = scores.squeeze(2).transpose(1, 0)  # [B*N, L]
        attn_weights = torch.softmax(scores, dim=1)
        outputs = outputs.permute(1, 0, 2)  # [B*N, L, H]
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        embed = torch.relu(self.bn1(self.linear1(attn_embed)))
        parameters = self.linear2(embed)
        # return parameters[:, 0], parameters[:, 1]   # mu, sigma
        return parameters.squeeze(-1)


if __name__ == "__main__":
    a = torch.randn((16, 20, 3))
    net = MSU(3, 20, 128)
    b = net(a)
    print(b.shape)