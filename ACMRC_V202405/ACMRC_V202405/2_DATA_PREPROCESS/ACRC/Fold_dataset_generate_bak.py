#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : data_shuffle.py   数据划分平衡（长度）
@Author  : huanggj
@Time    : 2022/12/12 18:50
"""
import os, json, copy, random
from transformers import AutoTokenizer, BertTokenizer
import io, json, codecs, os
import random
from tqdm import tqdm
ave = lambda x : sum(x)/len(x)
codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=3, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)
json_dumpsl = lambda d: json.dumps(d, ensure_ascii=False)


def get_data(original_data_path):
    dataset = json_load(original_data_path)


    dataset_list = []
    for item in tqdm(dataset['data']):
        dataset_list.append(item)

    return dataset_list

def divideIntoNstrand(listTemp, n):
    twoList = [ [] for i in range(n)]
    for i,e in enumerate(listTemp):
        twoList[i%n].append(e)
    return twoList

def get_d(data_list, cid_list):
    return_list = []
    for d in data_list:
        cid = d['cid']
        if cid in cid_list:
            return_list.append(d)

    return return_list

# 根据token长度分配数据集
def data_classify_by_length(total_list):
    # q1 选正确题型
    q1_512 = []
    q1_1024 = []
    q1_1536 = []
    q1_other = []
    # q2 选错误题型
    q0_512 = []
    q0_1024 = []
    q0_1536 = []
    q0_other = []

    q0_max_length = -1
    q1_max_length = -1
    for d in total_list:
        cid = d['cid']
        context = d['context']
        # 文章token长度
        length = len(tokenizer(context)["input_ids"])
        q_type = d['qas'][0]['question_type']

        # q1 选正确题型
        if q_type == 1:
            if length <= 512:
                q1_512.append(cid)
            elif length <= 1024:
                q1_1024.append(cid)
            elif length <= 1536:
                q1_1536.append(cid)
            else:
                q1_other.append(cid)
            if q1_max_length < length:
                q1_max_length = length

        # q2 选错误题型
        if q_type == 0:
            if length <= 512:
                q0_512.append(cid)
            elif length <= 1024:
                q0_1024.append(cid)
            elif length <= 1536:
                q0_1536.append(cid)
            else:
                q0_other.append(cid)

            if q0_max_length < length:
                q0_max_length = length

    print("##########  选正确题型汇总   #########")
    print("question type 1(选正确),  <= 512, count : {}".format(len(q1_512)))
    print("question type 1(选正确),  <= 1024, count : {}".format(len(q1_1024)))
    print("question type 1(选正确),  <= 1536, count : {}".format(len(q1_1536)))
    print("question type 1(选正确),  > 1536, count : {}".format(len(q1_other)))
    print("question type 1(选正确),  max length {}".format(q1_max_length))

    print("##########  选错误题型汇总   #########")
    print("question type 0(选错误),  <= 512, count : {}".format(len(q0_512)))
    print("question type 0(选错误),  <= 1024, count : {}".format(len(q0_1024)))
    print("question type 0(选错误),  <= 1536, count : {}".format(len(q0_1536)))
    print("question type 0(选错误),  > 1536, count : {}".format(len(q0_other)))
    print("question type 0(选错误),  max length {}".format(q0_max_length))

    # 先打乱cid
    random.shuffle(q0_512)
    random.shuffle(q0_1024)
    random.shuffle(q0_1536)
    random.shuffle(q1_512)
    random.shuffle(q1_1024)
    random.shuffle(q1_1536)

    # 因为数据集 train:dev:test = 8:1:1 所以每种情况平均分成10份，每个集合拿走自己那份即可
    # 1024-1536 + 1536-  一起算，因为后面的不多，没办法均分10份
    tq0_512_ = divideIntoNstrand(q0_512, 10)
    tq0_1024_ = divideIntoNstrand(q0_1024, 10)
    tq0_1536_ = divideIntoNstrand(q0_1536 + q0_other, 10)

    tq1_512_ = divideIntoNstrand(q1_512, 10)
    tq1_1024_ = divideIntoNstrand(q1_1024, 10)
    tq1_1536_ = divideIntoNstrand(q1_1536 + q1_other, 10)

    # train 取前8份
    train_cid_list = []
    for i in range(8):
        train_cid_list.extend(tq0_512_[i])
        train_cid_list.extend(tq0_1024_[i])
        train_cid_list.extend(tq0_1536_[i])
        train_cid_list.extend(tq1_512_[i])
        train_cid_list.extend(tq1_1024_[i])
        train_cid_list.extend(tq1_1536_[i])

    # dev 取第9份
    dev_cid_list = []
    dev_cid_list.extend(tq0_512_[8])
    dev_cid_list.extend(tq0_1024_[8])
    dev_cid_list.extend(tq0_1536_[8])
    dev_cid_list.extend(tq1_512_[8])
    dev_cid_list.extend(tq1_1024_[8])
    dev_cid_list.extend(tq1_1536_[8])

    # test 取第10份
    test_cid_list = []
    test_cid_list.extend(tq0_512_[9])
    test_cid_list.extend(tq0_1024_[9])
    test_cid_list.extend(tq0_1536_[9])
    test_cid_list.extend(tq1_512_[9])
    test_cid_list.extend(tq1_1024_[9])
    test_cid_list.extend(tq1_1536_[9])

    print("###  最终数据集长度  ###")
    print("train len : {}".format(len(train_cid_list)))
    print("dev len : {}".format(len(dev_cid_list)))
    print("test len : {}".format(len(test_cid_list)))

    return train_cid_list, dev_cid_list, test_cid_list



def write_json(data, name, path):
    dataset = {
        'version': name,
        'data': data
    }

    if os.path.exists(path):
        os.remove(path)

    json_dump(dataset, path)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('/disk2/huanggj/ACMRC_EXPERIMENT/pretrain/BERT')

    # 原数据
    path = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/temp_data/ACMRC_TOTAL_1.json'
    # 写出路径
    train_path = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/new_data_0430/acrc_train.json'
    dev_path = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/new_data_0430/acrc_dev.json'
    test_path = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/new_data_0430/acrc_test.json'
    # 读取
    file = open(path, 'r', encoding='UTF-8')
    total_list = json.load(file)['data']

    random.shuffle(total_list)

    # 根据长度对cid进行分组
    train_cids, dev_cids, test_cids = data_classify_by_length(total_list)

    # 根据CID组装数据集
    train_list = get_d(total_list, train_cids)
    dev_list = get_d(total_list, dev_cids)
    test_list = get_d(total_list, test_cids)
    print("aaaa")
    # 写出数据集
    write_json(train_list, "acmrc-train", train_path)
    write_json(dev_list, "acmrc-dev", dev_path)
    write_json(test_list, "acmrc-test", test_path)