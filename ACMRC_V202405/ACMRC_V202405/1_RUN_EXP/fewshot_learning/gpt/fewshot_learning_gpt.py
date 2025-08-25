#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : fewshot_learning.py
@Author  : huanggj
@Time    : 2024/3/13 21:49
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import openai
import os, json, time

openai.api_key = "sk-V0xSFK4DBBWZN8Bk5d77Ea3aE04544E18a7b1414A1347d70"
#openai.organization = "org-emihOJt9xajeNVbUyPPh4LG5"
openai.api_base = "https://www.jcapikey.com/v1"
example_path = "../example.json"
example_map = None


def read_json(json_path):
    data_list = json.load(open(json_path, 'r', encoding='UTF-8'))['data']
    dict_ = dict()
    for d in data_list:
        dict_[int(d['cid'])] = d
    return data_list,dict_

def get_completed_cids(path):

    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        # 读取文件内容
        lines = f.read().splitlines()
        cids = []
        for line in lines:
            arr = line.split("|")
            try:
                cids.append(int(arr[0]))
            except:
                print(line)
        return cids

def remove_common_elements(list1, list2):
    # 使用列表推导，过滤掉list1中在list2中也存在的元素
    result = [x for x in list1 if x not in list2]
    return result

# 写入json数据
def write_json(data, des_json_path):
    data_dict = dict()
    data_dict['version'] = "acmrc-data"
    data_dict['data'] = data

    if os.path.exists(des_json_path):
        os.remove(des_json_path)

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=3, sort_keys=False)

def get_answer_from_gpt(dynasty, cid, passage, question, options):
    example_passage, example_question, example_options, example_answer, example_solution = getExample(dynasty)

    system_set = "你是一个古文阅读理解模型,你要帮我解答多项选择阅读理解题目。首先我会给你一个例子进行学习（包含1篇古文，1个问题，4个选项，1个答案，1段答案解析），然后我会给需要你解答的问题，你需要参考例子的解题思路得到答案，你只需要从A、B、C、D中选择一个答案返回，不要回答其他问题。"

    example_set = f'''古文：{example_passage}
                      问题：{example_question}
                      选项：A.{example_options[0]} B.{example_options[1]} C.{example_options[2]} D.{example_options[3]}
                      答案：{example_answer}
                      解析：{example_solution}
                    '''
    assistant_response = "我已经学会了，请告诉我实际要解答的阅读理解题目。"

    user_request = f'''古文：{passage}
                      问题：{question}
                      选项：A.{options[0]} B.{options[1]} C.{options[2]} D.{options[3]}  
                    '''

    # 构建对话
    message = [
        {"role": "system", "content": system_set},
        {"role": "user", "content": example_set},
        {"role": "assistant", "content": assistant_response},
        {"role": "user", "content": user_request}
    ]

    response = None
    while True:
        try:
            # 发送聊天请求
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=0
            )
            break
        except Exception as e:
            # 打印异常信息,等待30s重新请求
            print("发生异常：", str(e))
            time.sleep(5)
            continue

    # 输出模型的回复
    answer = response['choices'][0].message["content"]


    return answer

def getExample(dynasty):
    global example_map
    if example_map is None:
        # 加载
        example_map = json.load(open(example_path, 'r', encoding='UTF-8'))
        print("aa")
    a = example_map
    example_list = example_map.get(dynasty)
    example = example_list[0]

    return example['context'], example['question'], example['options'], example['answer'], example['solution']


def fewshot_learning(source_path, log_path):

    completed_cids = get_completed_cids(log_path)
    print("已有长度: {}".format(len(completed_cids)))
    data_list, datas_map = read_json(source_path)
    cids = list(datas_map.keys())
    print("数据长度: {}".format(len(cids)))
    if len(completed_cids) > 0:
        cids = [x for x in cids if x not in completed_cids]
        print("需要执行的数据长度: {}".format(len(cids)))

    if len(cids) > 0:
        for index, cid in enumerate(cids):
            data  = datas_map.get(cid)
            answer = get_answer_from_gpt(data['dynasty-gpt'], cid, data['context'], data['qas'][0]['question'], data['qas'][0]['options'])
            answer.replace("\n", "")
            log_line = f"{cid}|{answer}"
            print(f"cid : {cid}, answer : {answer}")
            with open(log_path, "a", encoding="utf-8") as f:  # Open the file in append mode
                f.write(log_line + "\n")  # Write the response to the file



if __name__ == '__main__':


    test_source_path = "../acrc_test_20240314.json"
    test_log_path = "../log.txt"


    fewshot_learning(test_source_path, test_log_path)
