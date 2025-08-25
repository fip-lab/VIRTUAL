#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : dataset_length_stats.py
@Author  : huanggj
@Time    : 2023/2/12 23:50
"""
import os, json, copy
import random
from transformers import AutoTokenizer, BertTokenizer


def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list



def del_data(data_list, delete_list):
    for d in data_list:
        cid = d['cid']


def count_answers(data_list):
    answer_count = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for item in data_list:
        answer_count[item['qas'][0]['answer']] += 1
    return answer_count


def adjust_options(data_list, target_count):
    current_count = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    for item in data_list:
        qa = item['qas'][0]
        current_answer = qa['answer']
        if current_count[current_answer] < target_count[current_answer]:
            current_count[current_answer] += 1
        else:
            for answer in target_count:
                if current_count[answer] < target_count[answer]:
                    index_current = ord(current_answer) - ord('A')
                    index_target = ord(answer) - ord('A')
                    qa['options'][index_current], qa['options'][index_target] = qa['options'][index_target], qa['options'][index_current]
                    qa['trans_options'][index_current], qa['trans_options'][index_target] = qa['trans_options'][index_target], qa['trans_options'][index_current]
                    qa['correctness'][index_current], qa['correctness'][index_target] = qa['correctness'][index_target], qa['correctness'][index_current]
                    qa['answer'] = answer
                    current_count[answer] += 1
                    break
    return data_list


def count_answers_by_question_type(data_list):
    answer_count_by_question_type = {0: {'A': 0, 'B': 0, 'C': 0, 'D': 0},
                                     1: {'A': 0, 'B': 0, 'C': 0, 'D': 0}}
    for item in data_list:
        question_type = item['qas'][0]['question_type']
        answer = item['qas'][0]['answer']
        answer_count_by_question_type[question_type][answer] += 1
    return answer_count_by_question_type




def write_json(data, des_json_path):
    data_dict = dict()
    data_dict['version'] = "acmrc-data"
    data_dict['data'] = data

    if os.path.exists(des_json_path):
        os.remove(des_json_path)

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=3, sort_keys=False)

lable2id = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
        }


if __name__ == '__main__':


    res_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/base_data/acrc_train_rebalance.json"
    des_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/data_aug_0605/acrc_train_balanced.json"
    #读取
    data_list = json.load(open(res_json_path, 'r', encoding='UTF-8'))['data']

    # 计算每个答案应该有的基本个数和剩余部分
    # base_target_count = len(data_list) // 4
    # remainder = len(data_list) % 4
    # target_count = {chr(ord('A') + i): base_target_count + (1 if i < remainder else 0) for i in range(4)}

    # 调整选项和答案
    # 调整选项和答案
    print(count_answers(data_list))
    # data_list = adjust_options(data_list, target_count)
    #
    # # 随机打乱新列表的顺序
    # random.shuffle(data_list)
    #
    # # 输出结果
    # #print(data_list)
    # # 检查调整后的答案分布
    # print(count_answers(data_list))
    #
    # # 写入
    # write_json(data_list, des_json_path)
    #
    # # 统计每种question_type的answer分别有多少个
    # answer_count_by_question_type = count_answers_by_question_type(data_list)
    # print(answer_count_by_question_type)
