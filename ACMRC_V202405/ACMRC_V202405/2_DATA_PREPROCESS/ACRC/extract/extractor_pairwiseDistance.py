#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : extractor_cosineSim.py
@Author  : huanggj
@Time    : 2022/9/5 16:20
"""
from extract.extractor_base import ExtractionBase
import jieba
import numpy as np
import torch

class ExtractorCosineSim(ExtractionBase):
    TOP1_POLICY = 'top_1'
    TOPN_POLICY = 'top_n'

    # 初始化方法
    def __init__(self, policy):
        ExtractionBase.__init__(self)
        self.LOGGER.logger.warning("Extractor : CosineSim")
        self.policy = policy
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # 抽取入口:
    def extract(self, option_list, passage):
        # 参数检查
        if option_list is None or len(option_list) != 4:
            print(option_list)
            self.LOGGER.logger.warning('ERROR *** 选项个数错误 ***请检查')
            return passage
            #raise Exception

        if passage is None:
            self.LOGGER.logger.warning('ERROR *** 文章为None ***请检查')
            raise Exception

        for index in range(len(option_list)):
            option_list[index] = option_list[index].replace('A', '').replace('B', '').replace('C', '').replace('D', '')

        passage_sentence_list = [x for x in passage.split('。') if x]
        if len(passage_sentence_list) < 3:
            self.LOGGER.logger.warning("文章句子小于3句，直接返回原文")
            return passage

        # 抽取与选项
        similar_sentences_list_of_option_A = self.extract_single(option_list[0], passage_sentence_list, self.OPTION_A)

        similar_sentences_list_of_option_B = self.extract_single(option_list[1], passage_sentence_list, self.OPTION_B)

        similar_sentences_list_of_option_C = self.extract_single(option_list[2], passage_sentence_list, self.OPTION_C)

        similar_sentences_list_of_option_D = self.extract_single(option_list[3], passage_sentence_list, self.OPTION_D)

        # 合并list
        similar_sentences_list_of_option_A.extend(similar_sentences_list_of_option_B)
        similar_sentences_list_of_option_A.extend(similar_sentences_list_of_option_C)
        similar_sentences_list_of_option_A.extend(similar_sentences_list_of_option_D)

        # 把单个选项的相近结果合并成一个字符串
        all_similar_sentences_str = ''.join(similar_sentences_list_of_option_A)

        self.LOGGER.logger.warning(" ")
        self.LOGGER.logger.warning("###################### new question end ##########################")
        self.LOGGER.logger.warning(" ")
        return all_similar_sentences_str


    def extract_single(self, option, passage_sentence_list, option_name):
        sentence_list_length = len(passage_sentence_list)
        self.LOGGER.logger.warning("文章句子个数( %d ), 最大索引号( %d )" % (sentence_list_length, sentence_list_length - 1))
        # 获取text rank得分最大句子的前后两句
        if self.TOP1_POLICY == self.policy:
            max_similarity_score = -1
            max_similarity_score_index = 0
            for index in range(sentence_list_length):
                sentence = passage_sentence_list[index]
                sents_similarity_score = self.cosine_sim(option, sentence)
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

            self.LOGGER.logger.warning("选项( %s ): 最大cosine得分( %.2f), 句子索引号( %d ), 返回句子索引(%d, %d, %d)" % (
            option_name, max_similarity_score, max_similarity_score_index, return_index_tup[0], return_index_tup[1],
            return_index_tup[2]))
            return_sentence_list = [passage_sentence_list[return_index_tup[0]], passage_sentence_list[return_index_tup[1]],
                                    passage_sentence_list[return_index_tup[2]]]
            self.LOGGER.logger.warning("选项文字 : %s" % option)
            self.LOGGER.logger.warning("相关句子 : %s" % return_sentence_list)
            # return return_sentence_list, return_index_tup, passage_sentence_list
            return return_sentence_list

        return []



    def cosine_sim(self, sentence1, sentence2):
        cut1 = jieba.cut(sentence1)
        cut2 = jieba.cut(sentence2)

        list_word1 = (','.join(cut1)).split(',')
        list_word2 = (','.join(cut2)).split(',')

        # print(list_word1)
        # print(list_word2)

        # 取并集
        key_word = list(set(list_word1 + list_word2))
        #print(key_word)

        # 给定形状和类型的用0填充的矩阵存储向量
        word_vector1 = np.zeros(len(key_word))
        word_vector2 = np.zeros(len(key_word))

        # 依次确定向量的每个位置的值
        for i in range(len(key_word)):
            # 遍历key_word中每个词在句子中的出现次数
            for j in range(len(list_word1)):
                if key_word[i] == list_word1[j]:
                    word_vector1[i] += 1

            for k in range(len(list_word2)):
                if key_word[i] == list_word2[k]:
                    word_vector2[i] += 1

        # 输出向量
        #print(word_vector1)
        #print(word_vector2)
        # 返回

        output = self.cos(word_vector1, word_vector2)
        cosine_sim = float(np.sum(word_vector1 * word_vector2)) / (np.linalg.norm(word_vector1) * np.linalg.norm(word_vector2))
        #self.LOGGER.logger.warning("cosine sim : %.2f"%cosine_sim)
        return cosine_sim


