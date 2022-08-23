import torch
import os
import numpy as np


def train_with_valid(train_Dataloader, valid_Dataloader, num_epoch, net,
                     optimizer, tic_list, discriptor_path):
    if torch.cuda.is_available():
        net.cuda()
    valid_score_list = []
    for j in range(num_epoch):
        epoch_index = j
        # here is the train process for each epoch
        for X, y in train_Dataloader:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            for tic_number in range(len(tic_list)):
                input = X[:, tic_number, :]
                model = net
                output = model(input)
                if tic_number == 0:
                    output_all = output.unsqueeze(1)
                else:
                    output_new = output.unsqueeze(1)
                    output_all = torch.cat([output_all, output_new], dim=1)
            output_all = output_all.reshape(-1).unsqueeze(0)
            y = y.reshape(-1).unsqueeze(0)
            all = torch.cat([output_all, y], dim=0)
            cor = torch.corrcoef(all)[0, 1]
            loss = -cor
            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
        model_path = discriptor_path + "/all_model/" + "num_epoch_" + str(
            epoch_index + 1)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = model_path + "/" + "discroptor.pth"
        torch.save(net, model_path)

        valid_scores = np.array(1)
        for X, y in valid_Dataloader:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            for tic_number in range(len(tic_list)):
                input = X[:, tic_number, :]
                model = net
                output = model(input)
                if tic_number == 0:
                    output_all = output.unsqueeze(1)
                else:
                    output_new = output.unsqueeze(1)
                    output_all = torch.cat([output_all, output_new], dim=1)
            output_all = output_all.reshape(-1).unsqueeze(0)
            y = y.reshape(-1).unsqueeze(0)
            all = torch.cat([output_all, y], dim=0)
            cor = torch.corrcoef(all)[0, 1].cpu().detach().numpy()
            valid_scores = np.append(valid_scores, cor)
        valid_score = np.mean(valid_scores)
        valid_score_list.append(valid_score)
    best_model_index = valid_score_list.index(np.max(valid_score_list))
    model_path = discriptor_path + "/all_model/" + "num_epoch_" + str(
        best_model_index + 1) + "/" + "discroptor.pth"
    model = torch.load(model_path)
    best_model_path = discriptor_path + "/best_model/"
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    best_model_path = best_model_path + "discroptor.pth"
    torch.save(model, best_model_path)
