#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : passage_option_sim.py
@Author  : huanggj
@Time    : 2023/2/6 11:16
"""
import json, os, copy, re
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util

def get_split_list(text):
    """
    返回字符串分割后，不为空的元素
    @param text:
    @return:
    """
    question_text_list = re.findall('[^。]*。', text)
    return question_text_list


# 总体相似度
def overall_cos_sim(bert_embedder, option, passage_embed_list):
    bert_encoded_option = bert_embedder.encode(option, convert_to_tensor=True)
    total_score = 0
    for sent_embed in passage_embed_list:
        #sim = util.pytorch_cos_sim(bert_encoded_option, sent_embed)
        sim = cos_sim(bert_encoded_option, sent_embed).item()
        print(sim)
        total_score = total_score + sim
    _cos_sim = total_score/len(passage_list)

    print("总体相似度 : {}".format(_cos_sim))
    return _cos_sim

# 最大相似度
def max_cos_sim(bert_embedder, option, passage_embed_list):
    bert_encoded_option = bert_embedder.encode(option, convert_to_tensor=True)
    max_score = 0
    max_idx = -1
    for idx in range(len(passage_embed_list)):
        sent_embed = passage_embed_list[idx]
        sim = cos_sim(bert_encoded_option, sent_embed).item()
        if sim > max_score:
            max_score = sim
            max_idx = idx

    print("最大相似度 : {}，索引 : {}".format(max_score, max_idx))
    return max_score, max_idx

def get_passage_embed(passage_list):
    r_list = []
    for sent in passage_list:
        bert_encoded_sent = bert_embedder.encode(sent, convert_to_tensor=True)
        r_list.append(bert_encoded_sent)
    return r_list


if __name__ == '__main__':
    if os.path.exists('../data/external/distiluse-base-multilingual-cased-v1'):
        bert_embedder = SentenceTransformer('../data/external/distiluse-base-multilingual-cased-v1')
    else:
        bert_embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    res_json_path = "/disk2/huanggj/ACMRC_INTEGRATION/processed_data/current/micro_trans_option_29.json"
    des_json_path = "/disk2/huanggj/ACMRC_INTEGRATION/processed_data/current/trans_sim_3.json"

    new_list = []
    # 加载数据
    file = open(res_json_path, 'r', encoding='UTF-8')
    json_data = json.load(file)['data']
    index = 1
    for data_obj in json_data:
        print("current id {}".format(index))
        index = index + 1
        cid = data_obj['cid']
        passage_list = get_split_list(data_obj['context'])
        embed_passage = get_passage_embed(passage_list)
        option_list = data_obj['qas'][0]['options']
        tmp_obj = copy.deepcopy(data_obj)
        # 每一个选项与文章的整体相似度
        avg_score_list = []
        max_score_list = []
        for option in option_list:
            # 平均相似度
            avg_sim_socore = overall_cos_sim(bert_embedder, option, embed_passage)
            avg_score_list.append(avg_sim_socore)
            # 最大相似度
            max_sim_socore, max_idx = max_cos_sim(bert_embedder, option, embed_passage)
            max_score_list.append((max_sim_socore, max_idx))

        tmp_obj['qas'][0]['avg_sim'] = avg_score_list
        tmp_obj['qas'][0]['max_sim'] = max_score_list
        new_list.append(tmp_obj)

    data_dict = dict()
    data_dict['version'] = "acmrc-data-sim"
    data_dict['data'] = new_list

    if os.path.exists(des_json_path):
        os.remove(des_json_path)

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=3, sort_keys=False)
