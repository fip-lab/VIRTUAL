#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : dataset_length_stats.py
@desc    : 统计数据集中选项的个数
@Author  : huanggj
@Time    : 2023/2/12 23:50
"""
import json

def do_stat(path):
    data_list = json.load(open(path, 'r', encoding='UTF-8'))['data']
    answer_list = []
    for data in data_list:
        answer = data['qas'][0]['answer']
        answer_list.append(answer)

    a = answer_list.count("A")
    b = answer_list.count("B")
    c = answer_list.count("C")
    d = answer_list.count("D")

    print(path)
    print(f"A : {a} , B : {b} , C : {c} , D : {d} ")


if __name__ == '__main__':


    train_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/new_data_0430/acrc_dev_rebalance.json"
    dev_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/new_data_0430/acrc_dev_rebalance.json"
    test_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/new_data_0430/acrc_dev_rebalance.json"

    do_stat(train_path)
    do_stat(dev_path)
    do_stat(test_path)





