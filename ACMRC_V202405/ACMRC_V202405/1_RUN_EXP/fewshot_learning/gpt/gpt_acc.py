#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : add_dynasty.py
@Author  : huanggj
@Time    : 2024/3/14 21:20
"""

import os, json



def get_json(path):
    data_list = json.load(open(path, 'r', encoding='UTF-8'))['data']

    glm_map = dict()
    gpt_map = dict()

    for data in data_list:
        cid = int(data['cid'])
        glm_map[cid] = data['dynasty-gpt']
        gpt_map[cid] = data['dynasty-glm']

    return gpt_map, glm_map


def write_json(data, des_json_path):
    data_dict = dict()
    data_dict['version'] = "acmrc-data"
    data_dict['data'] = data

    if os.path.exists(des_json_path):
        os.remove(des_json_path)

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=3, sort_keys=False)

def get_answer_map(path):
    with open(path, "r", encoding="utf-8") as f:
        # 读取文件内容
        map_ = dict()
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split("|")
            cid = int(arr[0])
            answer = arr[1]
            map_[cid] = answer
        return map_


if __name__ == '__main__':

    test_source_path = "../acrc_test_20240314.json"
    log_path = "log_final.txt"


    answer_map = get_answer_map(log_path)

    data_list = json.load(open(test_source_path, 'r', encoding='UTF-8'))['data']


    correct_cnt = 0
    for data in data_list:
        cid = int(data['cid'])
        correct_answer = data['qas'][0]['answer']
        gpt_answer = answer_map.get(cid)
        if gpt_answer is None:
            print(cid)

        if gpt_answer != correct_answer:
            print("===================")
            print(cid)
            print(correct_answer)
            print(gpt_answer)
            print("===================")

        if gpt_answer == correct_answer:
            correct_cnt += 1

    print("correct_cnt: ", correct_cnt)
    print("total_cnt: ", len(data_list))
    print("accuracy: ", correct_cnt / len(data_list))