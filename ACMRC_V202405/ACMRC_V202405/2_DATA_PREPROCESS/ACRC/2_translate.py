#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : 2_translate.py
@Author  : huanggj
@Time    : 2023/2/2 20:31
"""
import os
import json
from copy import deepcopy
from translate.Baidu_Text_transAPI import baidu_trans
from translate.Microsoft_azure_transAPI import microsoft_trans



def get_translated_passage(cid):
    global trans_data
    if trans_data is None:
        path_ = "/disk2/huanggj/ACMRC_INTEGRATION/processed_data/pass/microsoft_trans.json"
        file = open(path_, 'r', encoding='UTF-8')
        trans_data = json.load(file)['data']

    for d_ in trans_data:
        if d_['cid'] == cid and 'trans_context' in d_:
            return True, d_['trans_context']

    return False, ''

def translate_data(json_data):
    # 遍历翻译
    new_data = []
    index = 1
    for data_obj in json_data:
        print("current id {}".format(index))
        index = index + 1
        cid = data_obj['cid']
        tmp_obj = deepcopy(data_obj)
        passage = data_obj['context']
        option_list = data_obj['qas'][0]['options']
        question = data_obj['qas'][0]['question']

        # 翻译文章
        #flag, trans_passage = get_translated_passage(cid)
        flag = False
        if not flag:
            print("文章需要翻译器翻译, cid : {}".format(cid))
            trans_passage = translator(passage, 'passage')
            print(trans_passage)

        if flag:
            print("文章已经翻译")
        tmp_obj['trans_context'] = trans_passage

        # 翻译选项
        new_options = []
        print("选项翻译, cid : {}".format(cid))
        for idx in range(4):
            tmp_option = translator(option_list[idx], 'option')
            print("cid: {}, 选项序号 : {}, 翻译后 : {}".format(cid, idx, tmp_option))
            new_options.append(tmp_option)
        tmp_obj['qas'][0]['trans_options'] = new_options

        # 翻译文章
        trans_question = translator(question, 'option')
        print(trans_question + '-cid-' + str(cid))
        tmp_obj['qas'][0]['trans_question'] = trans_question
        new_data.append(tmp_obj)

    return new_data

def write_json(data, des_json_path):
    data_dict = dict()
    data_dict['version'] = "acmrc-data"
    data_dict['data'] = data

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=3, sort_keys=False)




if __name__ == '__main__':
    baidu_api = 'baidu'
    microsoft_api = 'microsoft'


    res_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/temp_data/acrc_1_add.json"
    des_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/temp_data/acrc_2_add.json"

    # 选择翻译的api
    use_api = microsoft_api
    #use_api = baidu_api
    translator = microsoft_trans if use_api == microsoft_api else baidu_trans

    # 加载数据
    file = open(res_json_path, 'r', encoding='UTF-8')
    json_data = json.load(file)['data']
    # 翻译
    translated_data = translate_data(json_data)
    # 写入数据
    write_json(translated_data, des_json_path)


