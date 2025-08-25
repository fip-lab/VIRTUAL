#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : predict.py
@Author  : huanggj
@Time    : 2023/2/16 23:41
"""
import torch
import time
import csv
from utils.time_util import get_time_dif
from train import ACRCDataset, evaluate
from torch.utils.data import DataLoader, SequentialSampler,RandomSampler

def predictor(config, model, ids_generater):
    # 加载验证集上效果最好的模型
    if config.do_valid:
        model.load_state_dict(torch.load(config.save_path))

    # 测试数据
    config.logger.info("predicting ...")
    test_data = ids_generater.get_test_ids()
    test_dataset = ACRCDataset(test_data)

    # 顺序采样器
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=config.batch_size)
    start_time = time.time()

    test_acc, test_loss, test_report, test_confusion, predictions, _, cids = evaluate(
        config, model, test_dataloader, test=True, return_predictions=True
    )

    pred_labels = [config.id2label[pred] for pred in predictions]
    csv_path = f"predictions_{config.model_name}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'pred'])
        for cid, pred in zip(cids, pred_labels):
            writer.writerow([cid, pred])

    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

