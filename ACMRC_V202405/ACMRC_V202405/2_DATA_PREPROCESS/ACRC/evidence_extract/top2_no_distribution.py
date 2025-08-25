#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : topk_sim.py
@Author  : huanggj
@Time    : 2023/5/2 19:15
"""
import numpy as np
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel



"""
    top2句子及其上下句，并去重，剩下的不分配
"""





class Input_Preparation:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.top_sentences = {}
        self.option_embeddings = {}

    # 为每个选项选择与其最相似的2个句子
    def select_top_2_with_context(self, option_embedding, all_sentences_embeddings):
        cos_similarities = cosine_similarity(option_embedding.reshape(1, -1), all_sentences_embeddings)
        top_2_indices = cos_similarities.argsort()[0][-2:]

        selected_indices = set()
        for idx in top_2_indices:
            if idx == 0:
                selected_indices.update([0, 1, 2])
            elif idx == len(all_sentences_embeddings) - 1:
                selected_indices.update([idx - 2, idx - 1, idx])
            else:
                selected_indices.update([idx - 1, idx, idx + 1])

        return sorted(list(selected_indices))

    # 对所有句子进行编码
    def sentences_embeddings(self, sentences):
        embeddings = []
        for sentence in sentences:
            with torch.no_grad():
                inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
                inputs = inputs.to('cuda')  # Move the input tensors to the GPU
                outputs = self.model(**inputs)
                sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                embeddings.append(sentence_embedding)
        return embeddings

    def compute_option_embeddings(self, options):
        with torch.no_grad():
            for option in options:
                inputs = self.tokenizer(option, return_tensors='pt', padding=True, truncation=True)
                inputs = inputs.to('cuda')  # Move the input tensors to the GPU
                outputs = self.model(**inputs)
                option_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                self.option_embeddings[option] = option_embedding

    def prepare_inputs(self, options, all_sentences, question):
        inputs = {}
        input_list = []
        self.compute_option_embeddings(options)
        all_sentences_embeddings = self.sentences_embeddings(all_sentences)

        # 步骤1：为每个选项计算与其最相似的k个句子
        self.top_sentence_indices = {}
        for option in options:
            top_indices = self.select_top_2_with_context(option_embedding=self.option_embeddings[option],
                                            all_sentences_embeddings=all_sentences_embeddings)
            self.top_sentence_indices[option] = top_indices
            print(top_indices)

        for i, option in enumerate(options):

            # 对每个选项分支中的句子索引按照文章中的顺序进行排序
            all_sentence_indices = self.top_sentence_indices[option]
            all_sentence_indices.sort()

            # 根据排序后的索引从原文中提取对应的句子
            self.top_sentences[option] = [all_sentences[idx] for idx in all_sentence_indices]

            text = question + option + '<SEP>' + '。'.join(self.top_sentences[option])
            #text = '。'.join(self.top_sentences[option])
            inputs[option] = text
            input_list.append(text)
        return input_list

def write_json(data, des_json_path):

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=3, sort_keys=False)


import json,os
from concurrent.futures import ProcessPoolExecutor


def process_batch(batch_data):
    # 加载模型和分词器
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model = model.to('cuda')
    tokenizer.add_tokens(['<选正确>', '<选错误>'])
    model.resize_token_embeddings(len(tokenizer))

    input_preparation = Input_Preparation(tokenizer, model)

    batch_prepared_inputs = []

    print("当前进程号 : {}, 处理数据条数: {}".format(os.getpid(), len(batch_data)))


    curr_data = 1
    for data in batch_data:
        context = data['context']
        cid = data['cid']
        options = data['qas'][0]['options']
        question_type = data['qas'][0]['question_type']
        question = '<选正确>' if question_type == 1 else '<选错误>'
        # 切分句子
        sentences = context.split('。')
        #print("当前进程号 : {}, 处理数据cid: {}".format(os.getpid(), cid))
        # 使用 Input_Preparation 类准备输入
        prepared_inputs = input_preparation.prepare_inputs(options, sentences, question)
        print(prepared_inputs)
        data['qas'][0]['options_evidence'] = prepared_inputs
        batch_prepared_inputs.append(data)
        print("当前进程号 : {}, 处理数据进度: {}/{}".format(os.getpid(), curr_data, len(batch_data)))
        curr_data += 1
    return batch_prepared_inputs


def chunks(lst, n):
    """将列表 lst 分成大小为 n 的块。"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


set_ = 'test'

source_path = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/new_data_0504/acrc_{}_rebalance.json'.format(set_)
des_path = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/top2_nd/acrc_{}.json'.format(set_)

# 加载数据集
with open(source_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# 将数据集划分为小批量
max_workers=6

batch_size = (len(dataset['data']) // max_workers) + 1
batches = list(chunks(dataset['data'], batch_size))

# 使用多进程处理数据
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    prepared_inputs_list = list(executor.map(process_batch, batches))

# 将结果扁平化
prepared_inputs_list = [item for sublist in prepared_inputs_list for item in sublist]

write_json(prepared_inputs_list, des_path)
