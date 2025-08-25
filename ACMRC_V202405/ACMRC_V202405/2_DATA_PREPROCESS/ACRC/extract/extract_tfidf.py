#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : extract_tfidf.py
@Author  : huanggj
@Time    : 2022/12/9 20:02

TF-IDF
1.删除词频最高的的词（一般为人名），然后再用余弦相似度
2.根据选项高频词来选句子
"""
import io, json, codecs, os
import random
ave = lambda x : sum(x)/len(x)
codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)
json_dumpsl = lambda d: json.dumps(d, ensure_ascii=False)


from data_preprocess.extract.tfidf.TFIDF import TFIDF
from extractor_base import ExtractionBase
from tqdm import tqdm
import re
import json
import pickle as pk
import numpy as np
import jieba

def sent_tokenize(para):
    # para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    # para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    # para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    # para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    # para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    # para = para.split("\n")
    # ret = []
    # for s in para:
    #     if len(s) > 0:
    #         ret.append(s)
    # return ret
    passage_sentence_list = re.findall('[^。]*。', para)
    return passage_sentence_list


def replace_keyword(options, keyword):
    for idx, option in enumerate(options):
        options[idx] = options[idx].replace(keyword, '')

    return options

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def cosine_similarity(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

class Embedding:
    def __init__(self, word_list, vector_list):
        self.word2id = {}
        self.id2word = {}
        self.word2id['[UNK]'] = 0
        self.id2word[0] = '[UNK]'
        for i, w in enumerate(word_list):
            self.word2id[w] = i + 1
            self.id2word[i+1] = w
        avg_vector = np.array(vector_list[0])
        for i in range(1, len(vector_list)):
            avg_vector += np.array(vector_list[i])
        avg_vector /= len(vector_list)

        self.vector_list = [avg_vector] + vector_list

    # 定义了这个方法后，可以 p[] 像数组一样取值
    def __getitem__(self, word):
        if word not in self.word2id:
            return self.vector_list[0]
        else:
            return self.vector_list[self.word2id[word]]

class DenseEmbedder:
    def __init__(self, embedding, tokenizer, stop_list=[]):
        self.embedding = embedding
        self.stop_list = set(stop_list)
        self.tokenizer = tokenizer

    def remove_stop_words(self, sent):
        #  jieba 分词
        tokenized_sent = self.tokenizer(sent, cut_all=True)
        ret = []
        for w in tokenized_sent:
            if w in self.stop_list:
                continue
            ret.append(w)
        return sorted(list(set(ret)))

    def encode(self, sent, mode='average'):
        tokenized_sent_without_stopwords = self.remove_stop_words(sent)
        if len(tokenized_sent_without_stopwords) == 0:
            return self.embedding['[UNK]']
        encoded_words = []
        for w in tokenized_sent_without_stopwords:
            encoded_words.append(self.embedding[w])
        if mode == 'average':
            a = np.mean(encoded_words, 0)
            return a

    def get_embedding_vectors(self, sent):
        tokenized_sent_without_stopwords = self.remove_stop_words(sent)
        encoded_words = []
        for w in tokenized_sent_without_stopwords:
            encoded_words.append(self.embedding[w])
        return encoded_words

class ExtractorTFIDF(ExtractionBase):
    def __init__(self):
        ExtractionBase.__init__(self)
        self.tfidf = TFIDF()
        self.pattern = re.compile("[^\u4e00-\u9fa5]")
        self.tag_num = 5
        self.word_list = pk.load(open("/disk2/huanggj/ACMRC_INTEGRATION/data_preprocess/extract/word_vectors/sgns.merge.char_word-list", 'rb'))
        self.vector_list = pk.load(open("/disk2/huanggj/ACMRC_INTEGRATION/data_preprocess/extract/word_vectors/sgns.merge.char_vector-list", 'rb'))
        #self.simple_stop_list = []

        punctuations = "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。.,/?![]\"\\"

        # stop words
        self.simple_stop_list = [c for c in punctuations]
        for line in open('/disk2/huanggj/ACMRC_INTEGRATION/data_preprocess/extract/tfidf/stop_words.txt', 'r', encoding='utf-8'):
            self.simple_stop_list.append(line.strip())

        self.dense_embedder = DenseEmbedder(Embedding(self.word_list, self.vector_list), jieba.cut,self.simple_stop_list)
    def get_keyword_of_options(self, options):
        """
        删除所有选项中的高频词（通常为人名）
        :return:
        """
        # 删除空格与非中文字符
        options = ''.join(options)
        options = options.replace(' ', '')
        options = re.sub(self.pattern, '', options)
        # TF-IDF 抽取关键词
        keywords = self.tfidf.extract_tags(options, topK=self.tag_num, withWeight=True)
        return keywords

    def search_evidence(self, queries, context_sents, dense_encoded_sents):
        # 对于每个选项
        res_sentence = []
        res_index = []
        max_index = len(context_sents) -1
        for query in queries:
            #candidates = []
            ### first retrieval ###
            query_tokens = self.dense_embedder.remove_stop_words(query)
            first_step_dense_encoded_query = self.dense_embedder.encode(query)
            #
            first_step_results = []
            for i, dense_encoded_sent in enumerate(dense_encoded_sents):
                sim = cosine_similarity(first_step_dense_encoded_query, dense_encoded_sent)
                first_step_results.append([context_sents[i], sim, i])
            first_step_results = sorted(first_step_results, key=lambda x: x[1], reverse=True)

            index = first_step_results[0][2]
            key_sentences = ''
            if index == 0:
                key_sentences = context_sents[0] + context_sents[1] + context_sents[2]
                res_index.extend([0, 1, 2])
            elif index == max_index:
                key_sentences = context_sents[max_index - 2] + context_sents[max_index - 1] + context_sents[max_index]
                res_index.extend([max_index - 2, max_index - 1, max_index])
            else:
                key_sentences = context_sents[index - 1] + context_sents[index] + context_sents[index + 1]
                res_index.extend([index - 1, index, index + 1])

            res_sentence.append(key_sentences)

        # 去重排序
        res_index = list(set(res_index))
        res_index.sort()
        all_sentences = []
        for i in res_index:
            all_sentences.append(context_sents[i])

        return res_sentence, all_sentences

    def iteratively_retrieve_evidence(self, original_data_path, save_data_path):
        dataset = json_load(original_data_path)
        new_dataset = {
            'version': dataset['version'] + ' with key sentence extract',
            'data': []
        }

        for item in tqdm(dataset['data']):
            # 文章拆分成list
            context_sents = sent_tokenize(item['context'])
            # 用word2vec 进行编码
            dense_encoded_sents = [self.dense_embedder.encode(s) for s in context_sents]
            for j, qas in enumerate(item['qas']):
                options = qas['options']
                print("【删除高频词前】")
                print(options)
                # 取TF_IDF得分最高的词
                keywords = self.get_keyword_of_options(options)[0][0]
                print("【高频词】 %s"%keywords)
                # 在所有选项中删除该词
                options = replace_keyword(options, keywords)
                print("【删除高频词后】")
                print(options)
                # 余弦相似度计算与选项最相近的句子
                res_sentence, all_sentences = self.search_evidence(qas['options'], context_sents, dense_encoded_sents)
                item['qas'][j]['key_sentences'] = res_sentence
                item['qas'][j]['all_key_sentences'] = all_sentences
            new_dataset['data'].append(item)
        json_dump(new_dataset, save_data_path)


if __name__ == '__main__':
    extractor = ExtractorTFIDF()
    extractor.iteratively_retrieve_evidence('/disk2/huanggj/ACMRC_INTEGRATION/base/data/acmrc_train.json', '/disk2/huanggj/ACMRC_INTEGRATION/base/data/train_p.json')
    extractor.iteratively_retrieve_evidence('/disk2/huanggj/ACMRC_INTEGRATION/base/data/acmrc_dev.json', '/disk2/huanggj/ACMRC_INTEGRATION/base/data/dev_p.json')
    extractor.iteratively_retrieve_evidence('/disk2/huanggj/ACMRC_INTEGRATION/base/data/acmrc_test.json', '/disk2/huanggj/ACMRC_INTEGRATION/base/data/test_p.json')