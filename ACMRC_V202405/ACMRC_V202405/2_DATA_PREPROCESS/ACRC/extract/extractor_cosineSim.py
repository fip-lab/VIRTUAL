#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : extractor_cosineSim.py
@Author  : huanggj
@Time    : 2022/9/5 16:20
"""
import heapq
import re
from data_preprocess.extract.extractor_base import ExtractionBase
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer,BertModel, BertConfig
import jieba
import numpy as np
import torch


class ExtractorCosineSim(ExtractionBase):
    TOP1_POLICY = 'top_1'
    TOPN_POLICY = 'top_n'

    # 初始化方法
    def __init__(self, policy, use_bert):
        ExtractionBase.__init__(self)
        self.LOGGER.logger.warning("Extractor : CosineSim")
        self.policy = policy
        if use_bert:
            self.tokenizer = BertTokenizer.from_pretrained("../pretrain_language_models/bert")
            self.config = BertConfig.from_pretrained("../pretrain_language_models/bert")
            self.bert = BertModel.from_pretrained("../pretrain_language_models/bert", config=self.config)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.pdist = torch.nn.PairwiseDistance(p=2)
        self.use_bert = use_bert

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

        # 去除选项中的字母
        for index in range(len(option_list)):
            option_list[index] = option_list[index].replace('A', '').replace('B', '').replace('C', '').replace('D', '')

        passage_sentence_list = re.findall('[^。]*。', passage)
        #passage_sentence_list = [x for x in passage.split('。') if x]
        if len(passage_sentence_list) < 3:
            self.LOGGER.logger.warning("文章句子小于3句，直接返回原文")
            return passage

        # 抽取与选项
        similar_sentences_list_of_option_A, index_set_A = self.extract_single(option_list[0], passage_sentence_list, self.OPTION_A)

        similar_sentences_list_of_option_B, index_set_B = self.extract_single(option_list[1], passage_sentence_list, self.OPTION_B)

        similar_sentences_list_of_option_C, index_set_C = self.extract_single(option_list[2], passage_sentence_list, self.OPTION_C)

        similar_sentences_list_of_option_D, index_set_D = self.extract_single(option_list[3], passage_sentence_list, self.OPTION_D)

        # 合并list
        similar_sentences_list_of_option_A.extend(similar_sentences_list_of_option_B)
        similar_sentences_list_of_option_A.extend(similar_sentences_list_of_option_C)
        similar_sentences_list_of_option_A.extend(similar_sentences_list_of_option_D)

        option_a_sentences_str = option_list[0] + ''.join(similar_sentences_list_of_option_A) + '[SEP]'
        option_b_sentences_str = option_list[1] + ''.join(similar_sentences_list_of_option_B) + '[SEP]'
        option_c_sentences_str = option_list[2] + ''.join(similar_sentences_list_of_option_C) + '[SEP]'
        option_d_sentences_str = option_list[3] + ''.join(similar_sentences_list_of_option_D)

        # 把单个选项的相近结果合并成一个字符串
        #all_similar_sentences_str = ''.join(similar_sentences_list_of_option_A)
        all_similar_sentences_str = option_a_sentences_str + option_b_sentences_str + option_c_sentences_str + option_d_sentences_str

        extracted_str = ''
        index_set = sorted(set.union(index_set_A, index_set_B, index_set_C, index_set_D))
        for index in index_set:
            sentence = passage_sentence_list[index].strip()
            extracted_str = extracted_str + sentence

        other_sentences = []
        for index in range(len(passage_sentence_list)):
            if index not in index_set:
                other_sentences.append(passage_sentence_list[index])

        self.LOGGER.logger.warning(" ")
        self.LOGGER.logger.warning("###################### new question end ##########################")
        self.LOGGER.logger.warning(" ")
        return other_sentences, extracted_str


    def extract_single(self, option, passage_sentence_list, option_name):
        sentence_list_length = len(passage_sentence_list)
        self.LOGGER.logger.warning("文章句子个数( %d ), 最大索引号( %d )" % (sentence_list_length, sentence_list_length - 1))
        # top1 策略
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

            self.LOGGER.logger.warning("选项( %s ): 最大cosineSim得分( %.2f), 句子索引号( %d ), 返回句子索引(%d, %d, %d)" % (
            option_name, max_similarity_score, max_similarity_score_index, return_index_tup[0], return_index_tup[1],
            return_index_tup[2]))
            return_sentence_list = [passage_sentence_list[return_index_tup[0]], passage_sentence_list[return_index_tup[1]],
                                    passage_sentence_list[return_index_tup[2]]]
            self.LOGGER.logger.warning("选项文字 : %s" % option)
            self.LOGGER.logger.warning("相关句子 : %s" % return_sentence_list)
            # return return_sentence_list, return_index_tup, passage_sentence_list
            return return_sentence_list, set(return_index_tup)

        # topn策略
        if self.TOPN_POLICY == self.policy:
            sentence_score_list = []
            for index in range(sentence_list_length):
                sentence = passage_sentence_list[index]
                sents_similarity_score = round(self.cosine_sim(option, sentence),2)
                #self.LOGGER.logger.warning(sents_similarity_score)
                sentence_score_list.append(sents_similarity_score)

            max_num_index_list = list(map(sentence_score_list.index, heapq.nlargest(3, sentence_score_list)))

            self.LOGGER.logger.warning("选项( %s ): 最大cosine得分( %.2f, %.2f, %.2f), 返回句子索引(%d, %d, %d)" % (
                option_name, sentence_score_list[max_num_index_list[0]], sentence_score_list[max_num_index_list[1]], sentence_score_list[max_num_index_list[2]], max_num_index_list[0], max_num_index_list[1], max_num_index_list[2]))

            return_sentence_list = [passage_sentence_list[max_num_index_list[0]],
                                    passage_sentence_list[max_num_index_list[1]],
                                    passage_sentence_list[max_num_index_list[2]]]
            self.LOGGER.logger.warning("选项文字 : %s" % option)
            self.LOGGER.logger.warning("相关句子 : %s" % return_sentence_list)
            # return return_sentence_list, return_index_tup, passage_sentence_list
            return return_sentence_list

        return []



    def cosine_sim(self, sentence1, sentence2):
        cosine_sim = 0

        if self.use_bert:
            # 使用bert
            # 句子一
            # sentence1 = '[CLS]' + sentence1
            # sentence2 = '[CLS]' + sentence2
            sentence1_all_ids = self.tokenizer(sentence1)
            sentence1_ids = torch.tensor([sentence1_all_ids["input_ids"][1:]], dtype=torch.long)
            sentence1_type_ids = torch.tensor([sentence1_all_ids["token_type_ids"][1:]], dtype=torch.long)
            sentence1_attention_mask = torch.tensor([sentence1_all_ids["attention_mask"][1:]], dtype=torch.long)
            sentence1_representation = self.bert(input_ids=sentence1_ids, attention_mask=sentence1_type_ids,token_type_ids=sentence1_attention_mask).last_hidden_state

            # 句子二
            sentence2_all_ids = self.tokenizer(sentence2)
            sentence2_ids = torch.tensor([sentence2_all_ids["input_ids"][1:]], dtype=torch.long)
            sentence2_type_ids = torch.tensor([sentence2_all_ids["token_type_ids"][1:]], dtype=torch.long)
            sentence2_attention_mask = torch.tensor([sentence2_all_ids["attention_mask"][1:]], dtype=torch.long)
            sentence2_representation = self.bert(input_ids=sentence2_ids, attention_mask=sentence2_type_ids,token_type_ids=sentence2_attention_mask).last_hidden_state

            #print(sentence1_representation)
            #print(sentence1_representation[0][0].reshape(1,-1))

            #print(sentence2_representation[0][0])


            #  torch : 余弦相似度
            cosine_sim = self.cos(sentence1_representation[0][0].reshape(1,-1), sentence2_representation[0][0].reshape(1,-1)).item()
            #print(cosine_sim)
            #print(cosine_sim)
            # 欧氏距离
            # cosine_sim = self.pdist(v1, v2)
        else:
            # 词袋模型
            cut1 = jieba.cut(sentence1)
            cut2 = jieba.cut(sentence2)

            list_word1 = (','.join(cut1)).split(',')
            list_word2 = (','.join(cut2)).split(',')

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
                    #  直接计算 ： 余弦相似度
            cosine_sim = float(np.sum(word_vector1 * word_vector2)) / (np.linalg.norm(word_vector1) * np.linalg.norm(word_vector2))

            # v1 = torch.from_numpy(word_vector1)
            # v2 = torch.from_numpy(word_vector2)
            # v1 = v1.reshape(1,-1)
            # v2 = v2.reshape(1,-1)

            #  torch : 余弦相似度
            #cosine_sim = self.cos(v1, v2 )
            # 欧氏距离
            #cosine_sim = self.pdist(v1, v2)
            #self.LOGGER.logger.warning("cosine sim : %.2f"%cosine_sim)
        return cosine_sim


# pdist = torch.nn.PairwiseDistance(p=2)
# input1 = torch.randn(128)
# input2 = torch.randn(128)
# output = pdist(input1, input2)
# print(output)

