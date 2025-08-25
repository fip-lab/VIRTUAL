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

def get_split_list(text):
    """
    返回字符串分割后，不为空的元素
    @param text:
    @return:
    """
    question_text_list = re.findall('[^。]*。', text)
    return question_text_list

def extract_top1(option_name, option, passage_sentence_list, passage_embed_list):
    sentence_list_length = len(passage_sentence_list)
    bert_encoded_option = bert_embedder.encode(option, convert_to_tensor=True)
    max_similarity_score = -1
    max_similarity_score_index = 0
    for index in range(sentence_list_length):
        sent_embed = passage_embed_list[index]
        sents_similarity_score = cos_sim(bert_encoded_option, sent_embed).item()
        if sents_similarity_score > max_similarity_score:
            max_similarity_score = sents_similarity_score
            max_similarity_score_index = index

    return_index_tup = None
    if max_similarity_score_index == 0:
        return_index_tup = (0, 1, 2)
    elif max_similarity_score_index == sentence_list_length - 1:
        return_index_tup = (sentence_list_length - 3, sentence_list_length - 2, sentence_list_length - 1)
    else:
        return_index_tup = (
            max_similarity_score_index - 1, max_similarity_score_index, max_similarity_score_index + 1)

    print("选项( %s ): 最大cosineSim得分( %.2f), 句子索引号( %d ), 返回句子索引(%d, %d, %d)" % (
        option_name, max_similarity_score, max_similarity_score_index, return_index_tup[0], return_index_tup[1],
        return_index_tup[2]))

    return_sentence_list = [passage_sentence_list[return_index_tup[0]], passage_sentence_list[return_index_tup[1]],
                            passage_sentence_list[return_index_tup[2]]]
    print("选项文字 : %s" % option)
    print("相关句子 : %s" % return_sentence_list)
    # return return_sentence_list, return_index_tup, passage_sentence_list
    return return_sentence_list, set(return_index_tup)


def extract_topn(option_name, option, passage_sentence_list, passage_embed_list):
    sentence_list_length = len(passage_sentence_list)
    bert_encoded_option = bert_embedder.encode(option, convert_to_tensor=True)
    sentence_score_list = []
    for index in range(sentence_list_length):
        sent_embed = passage_embed_list[index]
        sents_similarity_score = cos_sim(bert_encoded_option, sent_embed).item()
        sentence_score_list.append(sents_similarity_score)

    max_num_index_list = list(map(sentence_score_list.index, heapq.nlargest(3, sentence_score_list)))

    print("选项( %s ): 最大cosine得分( %.2f, %.2f, %.2f), 返回句子索引(%d, %d, %d)" % (
        option_name, sentence_score_list[max_num_index_list[0]], sentence_score_list[max_num_index_list[1]],
        sentence_score_list[max_num_index_list[2]], max_num_index_list[0], max_num_index_list[1],
        max_num_index_list[2]))

    return_sentence_list = [passage_sentence_list[max_num_index_list[0]],
                            passage_sentence_list[max_num_index_list[1]],
                            passage_sentence_list[max_num_index_list[2]]]
    print("选项文字 : %s" % option)
    print("相关句子 : %s" % return_sentence_list)
    # return return_sentence_list, return_index_tup, passage_sentence_list
    return return_sentence_list, set(max_num_index_list)


def extract(data_list):
    train_new_list = []
    for d in data_list:
        cid = d['cid']
        print("############    cid : {} ###############".format(cid))
        context = d['context']
        option_list = d['qas'][0]['options']
        passage_list = get_split_list(context)
        embed_passage = get_passage_embed(passage_list)

        # 原选项抽取
        top1_all_idx_set = set()
        topn_all_idx_set = set()
        for idx in range(4):
            option = option_list[idx]
            top1_sentence_list, top1_idx_set = extract_top1(str(idx), option, passage_list, embed_passage)
            top1_all_idx_set = set.union(top1_all_idx_set, top1_idx_set)
            topn_sentence_list, topn_idx_list = extract_topn(str(idx), option, passage_list, embed_passage)
            topn_all_idx_set = set.union(topn_all_idx_set, topn_idx_list)

        top1_extracted_str = ''
        index_set_top1 = sorted(top1_all_idx_set)
        for index in index_set_top1:
            sentence = passage_list[index].strip()
            top1_extracted_str = top1_extracted_str + sentence

        topn_extracted_str = ''
        index_set_topn = sorted(topn_all_idx_set)
        for index in index_set_topn:
            sentence = passage_list[index].strip()
            topn_extracted_str = topn_extracted_str + sentence

        d['extract_context_t1'] = top1_extracted_str
        d['extract_context_tn'] = topn_extracted_str

        train_new_list.append(d)
    return train_new_list


def get_passage_embed(passage_list):
    r_list = []
    for sent in passage_list:
        bert_encoded_sent = bert_embedder.encode(sent, convert_to_tensor=True)
        r_list.append(bert_encoded_sent)
    return r_list

def write_json(data, des_json_path):
    data_dict = dict()
    data_dict['version'] = "acmrc-data"
    data_dict['data'] = data

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=3, sort_keys=False)

if __name__ == '__main__':
    if os.path.exists('../data/external/distiluse-base-multilingual-cased-v1'):
        bert_embedder = SentenceTransformer('../data/external/distiluse-base-multilingual-cased-v1')
    else:
        bert_embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    res_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/temp_data/acrc_3_add.json"
    des_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/temp_data/acrc_4_add.json"

    data_list = json.load(open(res_json_path, 'r', encoding='UTF-8'))['data']

    # 抽取
    new_data = extract(data_list)

    # 写入
    write_json(new_data, des_json_path)

