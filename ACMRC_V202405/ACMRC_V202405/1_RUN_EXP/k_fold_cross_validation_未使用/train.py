#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : train.py
@Author  : huanggj
@Time    : 2023/2/16 23:41
"""
import time
import torch
import numpy as np
from torch.nn import functional
from sklearn import metrics
from tqdm import tqdm
from utils.time_util import get_time_dif
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.optim import Adam

def trainer(config, model, ids_generater):
    start_time = time.time()
    # train data
    train_data = ids_generater.get_train_ids()
    train_dataset = ACRCDataset(train_data)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)
    # dev data
    dev_data = ids_generater.get_dev_ids()
    dev_dataset = ACRCDataset(dev_data)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=config.batch_size)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    dev_best_loss = float('inf')
    total_batch = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.epochs))
        model.train()
        for batch in tqdm(train_dataloader):
            model_inputs, labels, cids = [_.to(config.device) for _ in batch[0]], batch[1].to(config.device), batch[2].to(config.device)
            logits = model(model_inputs)
            loss = functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 验证
            if config.do_valid and total_batch != 0 and total_batch % 100 == 0 :
                dev_acc, dev_loss  = evaluate(config=config, model=model, dev_dataloader=dev_dataloader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2}, Val Loss: {2:>5.2},  Val Acc: {3:>6.2%},  Time: {4} {5}'
                print(msg.format(total_batch, loss.item(),  dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            # early stop
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    return model


def evaluate(config, model, dev_dataloader, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in dev_dataloader:
            model_inputs, labels, cids = [_.to(config.device) for _ in batch[0]], batch[1].to(config.device), batch[2].to(config.device)
            logits = model(model_inputs)
            loss = functional.cross_entropy(logits, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(logits.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)

    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.label_list, digits=config.num_label)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(dev_dataloader), report, confusion
    return acc, loss_total / len(dev_dataloader)


class ACRCDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        model_inputs = [torch.tensor(x) for x in item['model_inputs']]
        labels = torch.tensor(item['label'])
        cid = torch.tensor(item['cid'])
        return model_inputs, labels, cid
