import torch
import os
import numpy as np
from torch import nn


def train_with_valid(train_Dataloader, valid_Dataloader, num_epoch, net,
                     optimizer, net_path):
    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        net.cuda()
    valid_score_list = []
    for i in range(num_epoch):
        print(i)
        epoch_index = i
        # here is the train process for each epoch
        for X, y in train_Dataloader:
            print(X.shape)

            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            y_preidct = net(X)
            loss = criterion(y_preidct, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model_path = net_path + "/all_model/" + "num_epoch_" + str(
            epoch_index + 1)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = model_path + "/" + "LSTM.pth"
        torch.save(net, model_path)

        valid_scores = np.array(1)
        for X, y in valid_Dataloader:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            y_preidct = net(X)
            loss = criterion(y_preidct, y).detach().cpu().numpy()
            valid_scores = np.append(valid_scores, loss)
        valid_score = np.mean(valid_scores)
        valid_score_list.append(valid_score)
    best_model_index = valid_score_list.index(np.min(valid_score_list))
    model_path = net_path + "/all_model/" + "num_epoch_" + str(
        best_model_index + 1) + "/" + "LSTM.pth"
    model = torch.load(model_path)
    best_model_path = net_path + "/best_model/"
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    best_model_path = best_model_path + "LSTM.pth"
    torch.save(model, best_model_path)
