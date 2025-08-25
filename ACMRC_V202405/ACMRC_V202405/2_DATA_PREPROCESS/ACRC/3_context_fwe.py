#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : 4_context_extract.py   文章抽取
@Author  : huanggj
@Time    : 2023/2/11 21:31
"""
import heapq
import json, os, copy, re
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util


def fwe(text : str):
    fwe_list = ["而", "何", "乎", "乃", "其", "且", "若", "为", "焉", "也", "以", "因", "于", "与", "则", "者", "之"]
    for fw in fwe_list:
        text = text.replace(fw, "")
    return text

def context_fwe(json_data):
    # 遍历翻译
    new_data = []
    index = 1
    for data_obj in json_data:
        print("current id {}".format(index))
        index = index + 1
        tmp_obj = copy.deepcopy(data_obj)
        passage = data_obj['context']

        # 文章fwe
        passage_fwe =  fwe(passage)
        tmp_obj['fwe_context'] = passage_fwe

        new_data.append(tmp_obj)

    return new_data

def write_json(data, des_json_path):
    data_dict = dict()
    data_dict['version'] = "acmrc-data"
    data_dict['data'] = data

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=3, sort_keys=False)


if __name__ == '__main__':

    res_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/temp_data/ACMRC_TOTAL.json"
    des_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/temp_data/ACMRC_TOTAL_1.json"

    # 加载数据
    file = open(res_json_path, 'r', encoding='UTF-8')
    json_data = json.load(file)
    # fwe
    translated_data = context_fwe(json_data)
    # 写入数据
    write_json(translated_data, des_json_path)

