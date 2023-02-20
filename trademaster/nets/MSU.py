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
        scores = self.attn2(torch.tanh(self.attn1(torch.cat([outputs, H_n], dim=2))))  # [L, B*N, 1]
        scores = scores.squeeze(2).transpose(1, 0)  # [B*N, L]
        attn_weights = torch.softmax(scores, dim=1)
        outputs = outputs.permute(1, 0, 2)  # [B*N, L, H]
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        embed = torch.relu(self.bn1(self.linear1(attn_embed)))
        parameters = self.linear2(embed)
        # return parameters[:, 0], parameters[:, 1]   # mu, sigma
        return parameters.squeeze(-1)


if __name__ == '__main__':
    a = torch.randn((16, 20, 3))
    net = MSU(3, 20, 128)
    b = net(a)
    print(b)
