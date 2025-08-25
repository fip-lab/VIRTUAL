#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : dataset_stats.py
@desc    : 统计数据集各部分的token长度
@Author  : huanggj
@Time    : 2023/2/12 0:43
"""
import os, json
from transformers import AutoTokenizer, BertTokenizer

def fwe(text : str):
    for fw in fwe_list:
        text = text.replace(fw, "")
    return text

# 数据集各项长度统计
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('/disk2/huanggj/ACMRC_EXPERIMENT/pretrain/BERT')

    path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/temp_data/ACMRC_TOTAL_1.json"
    data_list = json.load(open(path, 'r', encoding='UTF-8'))['data']

    fwe_list = ["而", "何", "乎", "乃", "其", "且", "若", "为", "焉", "也", "以", "因", "于", "与", "则", "者","之"]

    context_List = []
    context_fwe_List = []
    context_cid_List = []
    trans_context_List = []
    trans_cid_context_List = []

    question_list = []
    trans_question_list = []

    context_token_length_list = []
    trans_context_token_length_list = []
    fwe_context_token_length_list = []

    question_token_length_list = []

    option_token_total_length_list = []
    option_token_length_list_list = []

    t_option_token_total_length_list = []
    t_option_token_length_list_list = []

    for data in data_list:
        cid =data['cid']
        print("cid  : {}".format(cid))
        # 文章
        context = data['context']
        trans_context = data['trans_context']
        fwe_context = data['fwe_context']
        # if context not in context_List :
        #     context_fwe_List.append(fwe(context))
        #     context_List.append(context)
        #     context_cid_List.append(cid)
        # trans_context = data['trans_context']
        # if trans_context not in trans_context_List:
        #     trans_context_List.append(trans_context)
        #     trans_cid_context_List.append(cid)

        context_fwe_List.append(fwe_context)
        context_List.append(context)
        context_cid_List.append(cid)

        trans_context_List.append(trans_context)
        trans_cid_context_List.append(cid)

        #  问题
        question = data['qas'][0]['question']
        trans_question = data['qas'][0]['trans_question']

        question_list.append(question)
        trans_question_list.append(trans_question)

        # 选项
        options = data['qas'][0]['options']
        trans_options = data['qas'][0]['trans_options']

        # 文章长度
        context_token_length = len(tokenizer(context)["input_ids"])
        print("文章长度 : {}".format(context_token_length))

        # 翻译文章长度
        trans_context_token_length = len(tokenizer(trans_context)["input_ids"])
        print("翻译文章长度 : {}".format(trans_context_token_length))

        # fwe文章长度
        fwe_context_token_length = len(tokenizer(fwe_context)["input_ids"])
        print("fwe文章长度 : {}".format(fwe_context_token_length))

        # 问题长度
        question_token_length = len(tokenizer(question)["input_ids"])
        print("问题长度 : {}".format(question_token_length))

        # 原选项
        option_total_length = 0
        option_token_length_list = []
        for option in options:
            token_length = len(tokenizer(option)["input_ids"])
            print("option token length : {}".format(token_length))
            option_total_length = option_total_length + token_length
            option_token_length_list.append(token_length)

        print("#total#  option token length : {}".format(option_total_length))

        # 翻译选项
        trans_option_total_length = 0
        trans_option_token_length_list = []
        for option in trans_options:
            token_length = len(tokenizer(option)["input_ids"])
            print("trans option token length : {}".format(token_length))
            trans_option_total_length = trans_option_total_length + token_length
            trans_option_token_length_list.append(token_length)

        print("#total#  trans option token length : {}".format(trans_option_total_length))

        # 文章
        context_token_length_list.append(context_token_length)
        trans_context_token_length_list.append(trans_context_token_length)
        fwe_context_token_length_list.append(fwe_context_token_length)

        # 问题
        question_token_length_list.append(question_token_length)

        # 选项
        option_token_total_length_list.append(option_total_length)
        option_token_length_list_list.append(option_token_length_list)

        # 翻译选项
        t_option_token_total_length_list.append(trans_option_total_length)
        t_option_token_length_list_list.append(trans_option_token_length_list)

    print("############################################")
    print("total : {}".format(len(context_token_length_list)))
    print("total context set : {}".format(len(context_List)))
    print("total trans context set : {}".format(len(trans_context_List)))
    print(set(context_cid_List).difference(set(trans_cid_context_List)))
    print("#######")

    #文章长度
    total_c_length = 0
    for c in context_List:
        context_token_length = len(tokenizer(c)["input_ids"])
        print("context token length : {}".format(context_token_length))
        total_c_length = total_c_length + context_token_length

    avg_len = total_c_length / len(context_List)
    print("avg context length : {}".format(avg_len))

    # passage fwe
    total_c_fwe_length = 0
    for c in context_fwe_List:
        context_ids = tokenizer(c)["input_ids"]
        context_token_length = len(context_ids)
        #print("context token length : {}".format(context_token_length))
        total_c_fwe_length = total_c_fwe_length + context_token_length

    avg_len_fwe = total_c_fwe_length / len(context_fwe_List)
    print("avg fwe context length : {}".format(avg_len_fwe))
    #print("total fwe : {}".format(len(context_fwe_List)))

    # 翻译文章
    total_t_c_length = 0
    for c in trans_context_List:
        context_ids = tokenizer(c)["input_ids"]
        context_token_length = len(context_ids)
        total_t_c_length = total_t_c_length + context_token_length

    avg_t_len = total_t_c_length / len(trans_context_List)
    print("avg translate context length : {}".format(avg_t_len))

    #问题长度
    total_q_length = 0
    for q in question_list:
        context_ids = tokenizer(q)["input_ids"]
        context_token_length = len(context_ids)
        #print("context token length : {}".format(context_token_length))
        total_q_length = total_q_length + context_token_length

    avg_q_len = total_q_length / len(question_list)
    print("avg question length : {}".format(avg_q_len))
    print("total : {}".format(len(question_list)))

    #翻译问题长度
    # t_total_q_length = 0
    # for q in trans_question_list:
    #     context_ids = tokenizer(q)["input_ids"]
    #     context_token_length = len(context_ids)
    #     #print("context token length : {}".format(context_token_length))
    #     t_total_q_length = t_total_q_length + context_token_length
    #
    # avg_t_q_len = t_total_q_length / len(trans_question_list)
    # print("avg q length : {}".format(avg_t_q_len))
    # print("total trans question : {}".format(len(trans_question_list)))

    # 选项
    avg = sum(option_token_total_length_list)/(len(option_token_total_length_list) * 4)
    print("avg option length : {}".format(avg))
    print("total options: {}".format(len(option_token_total_length_list)))
    #t_option_token_total_length_list

    avg = sum(t_option_token_total_length_list)/(len(t_option_token_total_length_list) * 4)
    print("avg t_options length : {}".format(avg))
    print("total trasn options: {}".format(len(t_option_token_total_length_list)))