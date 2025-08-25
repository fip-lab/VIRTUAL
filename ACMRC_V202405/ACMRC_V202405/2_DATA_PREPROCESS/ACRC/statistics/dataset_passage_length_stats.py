#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : dataset_length_stats.py
@desc    : 统计多少个512能覆盖所有的输入
@Author  : huanggj
@Time    : 2023/2/12 23:50
"""
import os, json, copy, math, pandas
from transformers import AutoTokenizer, BertTokenizer

def fwe(text : str):
    for fw in fwe_list:
        text = text.replace(fw, "")
    return text

def stat(data_list, name):
    print("############")
    print(name)

    data_set = set(data_list)
    print(" 总个数 {}".format(len(data_list)))
    for n in data_set:
        k = data_list.count(n)
        print(" {} : 个数 {}".format(n, k))



# 这个是统计多少个512能覆盖所有的输入
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('../pretrain/bert')

    path = "/disk2/huanggj/ACMRC_INTEGRATION/processed_data/current/remove_dupdata_10.json"
    data_list = json.load(open(path, 'r', encoding='UTF-8'))['data']

    fwe_list = ["而", "何", "乎", "乃", "其", "且", "若", "为", "焉", "也", "以", "因", "于", "与", "则", "者", "之"]

    set_1 = []
    set_2 = []
    set_3 = []
    set_4 = []
    set_5 = []
    set_6 = []
    set_7 = []
    set_8 = []


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
        o_o = question_token_length + option_total_token_length
        t_o = trans_question_token_length + option_total_token_length
        o_t = question_token_length + trans_option_total_token_length
        t_t = trans_question_token_length + trans_option_total_token_length

        # ori_passge
        o_o_n = context_token_length /( 512 - o_o)
        t_o_n = context_token_length /( 512 - t_o)
        o_t_n = context_token_length /( 512 - o_t)
        t_t_n = context_token_length /( 512 - t_t)

        set_1.append(math.ceil( o_o_n ))
        set_2.append(math.ceil( t_o_n ))
        set_3.append(math.ceil( o_t_n ))
        set_4.append(math.ceil( t_t_n ))


        # trans_passage
        o_o_n_t = trans_context_token_length /( 512 - o_o)
        t_o_n_t = trans_context_token_length /( 512 - t_o)
        o_t_n_t = trans_context_token_length /( 512 - o_t)
        t_t_n_t = trans_context_token_length /( 512 - t_t)

        set_5.append(math.ceil( o_o_n_t ))
        set_6.append(math.ceil( t_o_n_t ))
        set_7.append(math.ceil( o_t_n_t ))
        set_8.append(math.ceil( t_t_n_t ))


    print("##############################################")
    print("##############################################")

    stat(set_1, "o_o_o_list")
    stat(set_2, "o_t_o_list")
    stat(set_3, "o_o_t_list")
    stat(set_4, "o_t_t_list")
    stat(set_5, "t_o_o_list")
    stat(set_6, "t_t_o_list")
    stat(set_7, "t_o_t_list")
    stat(set_8, "t_t_t_list")