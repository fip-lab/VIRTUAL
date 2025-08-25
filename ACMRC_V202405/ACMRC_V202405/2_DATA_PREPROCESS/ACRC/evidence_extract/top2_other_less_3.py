#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : test1.py
@Author  : huanggj
@Time    : 2023/6/8 10:07
"""
#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : top2_test.py

top2及其上下句，在剩下的句子当中找到与其他选项余弦相似度最低的三句，

@Author  : huanggj
@Time    : 2023/6/6 21:07
"""
import jieba
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def split_sentences(text):
    # 利用标点符号进行句子切分
    return [s for s in text.split('。') if s]

def find_neighboring_sentences(sentences, index):
    # 检查索引并返回合适的句子
    if index == 0:
        return sentences[:3]
    elif index == len(sentences) - 1:
        return sentences[-3:]
    else:
        neighbors = []
        if index - 1 >= 0:
            neighbors.append(sentences[index - 1])
        neighbors.append(sentences[index])
        if index + 1 < len(sentences):
            neighbors.append(sentences[index + 1])
        return neighbors



def find_least_similar_sentences(options, sentences, vectorizer, used_sentences):
    # 去除已经选择过的句子
    unused_sentences = [sentence for sentence in sentences if sentence not in used_sentences]

    # 如果没有剩余句子，直接返回空列表
    if not unused_sentences:
        return []

    options_vector = vectorizer.transform([' '.join(jieba.cut(option)) for option in options])
    sentence_vectors = vectorizer.transform([' '.join(jieba.cut(sentence)) for sentence in unused_sentences])

    # 计算每个句子与所有选项的最大相似度
    max_similarities = []
    for sentence_vector in sentence_vectors:
        similarities = cosine_similarity(sentence_vector, options_vector)
        max_similarities.append(similarities.max())

    # 找出最大相似度最低的三个句子的索引
    bottom_three_indices = np.array(max_similarities).argsort()[:3]

    return [unused_sentences[i] for i in bottom_three_indices]


def find_similar_and_least_similar_sentences(text, options):
    sentences = split_sentences(text)
    results = {}

    vectorizer = TfidfVectorizer(tokenizer=jieba.cut)
    sentence_vectors = vectorizer.fit_transform(sentences)

    for option in options:
        option_vector = vectorizer.transform([' '.join(jieba.cut(option))])
        similarities = cosine_similarity(option_vector, sentence_vectors)

        # 找出余弦相似度最高的两个句子的索引
        top_two = similarities[0].argsort()[-2:][::-1]

        # 为每个选项找出最相关的两个句子，以及每个句子的上下句
        top_sentences = [find_neighboring_sentences(sentences, index) for index in top_two]

        # 找出和其他选项余弦相似度最低的三个句子
        all_top_sentences = [sentence for group in top_sentences for sentence in group]
        least_similar_sentences = find_least_similar_sentences(options, sentences, vectorizer, all_top_sentences)

        # 去重并排序
        all_sentences = all_top_sentences + least_similar_sentences
        unique_sentences = list(dict.fromkeys(all_sentences))  # 去重但保持顺序
        unique_sentences.sort(key=sentences.index)  # 按在原文中的顺序排序

        results[option] = unique_sentences

    return results




def write_json(data, des_json_path):
    data_dict = dict()
    data_dict['version'] = "acmrc-data"
    data_dict['data'] = data

    if os.path.exists(des_json_path):
        os.remove(des_json_path)

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=3, sort_keys=False)



if __name__ == '__main__':

    set_ = 'test'

    res_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/4_input_0606/acrc_{}1.json".format(set_)
    des_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/4_input_0606/acrc_{}.json".format(set_)
    # 读取
    data_list = json.load(open(res_json_path, 'r', encoding='UTF-8'))['data']

    # 数据增强
    final_list = []
    for data in data_list:
        print(data['cid'])
        options = data['qas'][0]['options']
        question_type = data['qas'][0]['question_type']
        question = '<选正确>' if question_type == 1 else '<选错误>'
        options_evidence = data['qas'][0]['options_evidence']

        for idx in range(4):
            print(len(options_evidence))
            evi = options_evidence[idx]
            option = options[idx]
            options_evidence[idx] = question + option + '<SEP>' + evi
            #print(options_evidence[idx])
        data['qas'][0]['options_evidence'] = options_evidence
        final_list.append(data)


    # 写入
    write_json(final_list, des_json_path)




