#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : extractor_word2vec.py
@Author  : huanggj
@Time    : 2022/8/6 18:59
"""
from sklearn.metrics.pairwise import cosine_similarity
#m和c向量表示的余弦相似度，word2vec生成的单词向量平均得到
from gensim.models import Word2Vec
#import word2vec.Domain_Spacy_tool.Domain_Token_Spacy as tool
import numpy as np

# 对每个句子的所有词向量取均值，来生成一个句子的vector
# sentence是输入的句子，size是词向量维度，w2v_model是训练好的词向量模型
def build_sentence_vector(sentence, size, w2v_model):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in sentence:
        try:
            vec += w2v_model[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算两个句向量的余弦相似性值
def cosine_similarity(vec1, vec2):
    a = np.array(vec1)
    b = np.array(vec2)
    cos1 = np.sum(a * b)
    cos21 = np.sqrt(sum(a ** 2))
    cos22 = np.sqrt(sum(b ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value


# 输入两个句子，计算两个句子的余弦相似性
def compute_cosine_similarity(sents_1, sents_2):
    size = 300
    w2v_model = Word2Vec.load('w2v_model.pkl')
    vec1 = build_sentence_vector(sents_1, size, w2v_model)
    vec2 = build_sentence_vector(sents_2, size, w2v_model)
    similarity = cosine_similarity(vec1, vec2)
    return similarity