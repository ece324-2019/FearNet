import torch
import torch.nn as nn

def accuracy(predictions, label):
    # print(label.size())
    # sig = nn.Sigmoid()
    # predictions = sig(predictions)
    predictions = predictions.detach()
    total_corr = 0
    index = 0
    for pred in predictions:
        p_val,p_clas = torch.max(nn.Softmax(pred),0)
        v_val,v_clas = torch.max(label[index],0)
        if p_clas.item() == v_clas.item():
            total_corr += 1
        index += 1
    return [(total_corr / len(label)),total_corr]

def evaluate(model, val_loader):
    opt = nn.CrossEntropyLoss()
    # opt = nn.MSELoss()
    valacc = 0
    loss_accum = 0
    idx = 0
    for j, batch in enumerate(val_loader,1):
        idx += 1
        valid_train, valid_label = batch
        predict = model(valid_train.float())
        loss = opt(predict.squeeze(), torch.max(valid_label,1)[1])
        # loss = opt(predict.squeeze(),valid_label.float())
        loss_accum += loss.item()
        valacc += accuracy(predict,valid_label)[0]
    return [valacc/idx,loss_accum/idx]
