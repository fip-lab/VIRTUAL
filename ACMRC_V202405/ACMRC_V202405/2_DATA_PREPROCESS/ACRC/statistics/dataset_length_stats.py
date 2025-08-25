#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : dataset_length_stats.py
@desc    : 统计输入长度分布
@Author  : huanggj
@Time    : 2023/2/12 23:50
"""
import os, json, copy
from transformers import AutoTokenizer, BertTokenizer

def fwe(text : str):
    for fw in fwe_list:
        text = text.replace(fw, "")
    return text

def stat(data_list, name):
    print("############")
    print(name)
    c_512 = []
    c_1024 = []
    c_1536 = []
    for d in data_list:
        if  d <= 512:
            c_512.append(d)
        if  d <= 1024:
            c_1024.append(d)
        if  d <= 1536:
            c_1536.append(d)

    print("512 count: {}".format(len(c_512)))
    print("1024 count: {}".format(len(c_1024)))
    print("1500 count: {}".format(len(c_1536)))
    print("total count: {}".format(len(data_list)))


# 输入长度分布
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('/disk2/huanggj/ACMRC_EXPERIMENT/pretrain/BERT')

    path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/temp_data/ACMRC_TOTAL_1.json"
    data_list = json.load(open(path, 'r', encoding='UTF-8'))['data']

    fwe_list = ["而", "何", "乎", "乃", "其", "且", "若", "为", "焉", "也", "以", "因", "于", "与", "则", "者", "之"]

    o_o_o_list = []
    t_o_o_list = []
    o_t_o_list = []
    o_o_t_list = []
    f_o_o_list = []
    t_o_t_list = []
    o_t_t_list = []

    for data in data_list:
        cid =data['cid']
        print("cid  : {}".format(cid))
        #context 文章tokens
        context = data['context']
        trans_context = data['trans_context']
        context_fwe = fwe(context)

        context_ids = tokenizer(context)["input_ids"]
        context_token_length = len(context_ids)
        print("context token length : {}".format(context_token_length))
        trans_context_ids = tokenizer(trans_context)["input_ids"]
        trans_context_token_length = len(trans_context_ids)
        print("trans context token length : {}".format(trans_context_token_length))
        context_fwe_ids = tokenizer(context_fwe)["input_ids"]
        context_fwe_token_length = len(context_fwe_ids)
        print("context fwe token length : {}".format(context_fwe_token_length))


        # question
        question = data['qas'][0]['question']
        trans_question = data['qas'][0]['trans_question']

        question_ids = tokenizer(question)["input_ids"]
        question_token_length = len(question_ids)
        print("question token length : {}".format(question_token_length))

        trans_question_ids = tokenizer(trans_question)["input_ids"]
        trans_question_token_length = len(trans_question_ids)
        print("question token length : {}".format(trans_question_token_length))


        # options
        options = data['qas'][0]['options']
        trans_options = data['qas'][0]['trans_options']


        # 原选项
        option_total_token_length = 0
        for option in options:
            option_ids = tokenizer(option)["input_ids"]
            token_length = len(option_ids)
            option_total_token_length = option_total_token_length + token_length
            #print("option token length : {}".format(token_length))
        print("all option token length : {}".format(option_total_token_length))

        # 翻译选项
        trans_option_total_token_length = 0
        for option in trans_options:
            option_ids = tokenizer(option)["input_ids"]
            token_length = len(option_ids)
            trans_option_total_token_length = trans_option_total_token_length + token_length
            #print("trans option token length : {}".format(token_length))
        print("all trans option token length : {}".format(trans_option_total_token_length))


        # 组合不同情况下的长度 o- original  t- translate  f - fwe
        o_o_o = context_token_length + question_token_length + option_total_token_length

        t_o_o = trans_context_token_length + question_token_length + option_total_token_length

        o_t_o = context_token_length + trans_question_token_length + option_total_token_length

        o_o_t = context_token_length + question_token_length + trans_option_total_token_length

        f_o_o = context_fwe_token_length + question_token_length + option_total_token_length

        t_o_t = trans_context_token_length + question_token_length + trans_option_total_token_length

        o_t_t = context_token_length + trans_question_token_length + trans_option_total_token_length

        o_o_o_list.append(o_o_o)
        t_o_o_list.append(t_o_o)
        o_t_o_list.append(o_t_o)
        o_o_t_list.append(o_o_t)
        f_o_o_list.append(f_o_o)
        t_o_t_list.append(t_o_t)
        o_t_t_list.append(o_t_t)

    print("##############################################")
    print("##############################################")

    stat(o_o_o_list, "o_o_o_list")
    stat(t_o_o_list, "t_o_o_list")
    stat(o_t_o_list, "o_t_o_list")
    stat(o_o_t_list, "o_o_t_list")
    stat(f_o_o_list, "f_o_o_list")
    stat(t_o_t_list, "t_o_t_list")
    stat(o_t_t_list, "o_t_t_list")