#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : data_loader_multiprocess.py
@Author  : huanggj
@Time    : 2023/6/13 20:52
"""

import torch
import json,os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer, BertTokenizer

def segment_init(tokenizer, max_length, input_type, example):
    question = example.question
    options = example.options
    # 先将question和options进行编码
    encoded_question_options = tokenizer('[CLS]' + question + '[SEP]' + '[SEP]'.join(options), truncation=True, padding=False, add_special_tokens=False)
    segments = []

    if input_type == '1':
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)

    elif input_type == '2_1':
        # segment 1
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)
        # segment 2
        segment = {}
        segment['input_ids'] = []
        segment['attention_mask'] = []
        segments.append(segment)

    elif input_type == '2_2':
        # segment 1
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)
        # segment 2
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)

    elif input_type == '3_1':
        # segment 1
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)
        # segment 2
        segment = {}
        segment['input_ids'] = []
        segment['attention_mask'] = []
        segments.append(segment)
        # segment 3
        segment = {}
        segment['input_ids'] = []
        segment['attention_mask'] = []
        segments.append(segment)

    elif input_type == '3_2':
        # segment 1
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)
        # segment 2
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)
        # segment 3
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)

    elif input_type == '5':
        # segment 1
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)
        # segment 2
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)
        # segment 3
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)
        # segment 4
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)
        # segment 5
        segment = {}
        segment['input_ids'] = encoded_question_options['input_ids']
        segment['attention_mask'] = encoded_question_options['attention_mask']
        segments.append(segment)

    elif input_type == '4':
        options_evidence = example.options_evidence
        for evidence in options_evidence:
            encoded_evidence = tokenizer(evidence,padding="max_length", max_length=max_length, truncation=True, add_special_tokens=True)
            segment = {}
            segment['input_ids'] = encoded_evidence['input_ids']
            segment['attention_mask'] = encoded_evidence['attention_mask']
            segments.append(segment)

    return segments


def prepare_input(tokenizer, max_length, input_type, example):


    if tokenizer.sep_token_id  == None:
        tokenizer.sep_token_id = 0
        print(tokenizer.sep_token_id)
    # 对passage按句子进行分割
    encoded_sentences = None
    if input_type != '4':
        sentences = example.passage.split('。')

        # 对每个句子进行编码，并添加到一个list中
        encoded_sentences = []
        for sentence in sentences:
            encoded_sentence = tokenizer( sentence + '。')
            encoded_sentences.append(encoded_sentence)


    # 分支初始化
    segments = segment_init(tokenizer, max_length, input_type, example)
    # 一个个分支去处理，一个填满了再填下一个
    if input_type != '4':
        for encoded_sentence in encoded_sentences:
            for segment in segments:
                # 检查当前分支是否可以添加这个句子，如果可以，就添加
                if len(segment['input_ids']) + len(encoded_sentence['input_ids']) <= max_length:
                    segment['input_ids'].extend(encoded_sentence['input_ids'])
                    segment['attention_mask'].extend(encoded_sentence['attention_mask'])
                    break
                else:
                    if len(segment['input_ids']) < max_length:
                        # 如果不能，就将当前分支填充到max_length，并处理下一个分支
                        padding_length = max_length - len(segment['input_ids']) - 1  # 留一个位置给[SEP]
                        segment['input_ids'].extend([0] * padding_length)
                        segment['input_ids'].append(tokenizer.sep_token_id)  # 添加[SEP]
                        segment['attention_mask'].extend([0] * padding_length)
                        segment['attention_mask'].append(1)  # 对于[SEP]的attention_mask为1

        # 对于每个还没有填充完的分支，在最后添加[SEP]
        for segment in segments:
            if len(segment['input_ids']) < max_length:
                padding_length = max_length - len(segment['input_ids']) - 1  # 留一个位置给[SEP]
                segment['input_ids'].extend([0] * padding_length)
                segment['input_ids'].append(tokenizer.sep_token_id)  # 添加[SEP]
                segment['attention_mask'].extend([0] * padding_length)
                segment['attention_mask'].append(1)  # 对于[SEP]的attention_mask为1


    result = []
    for segment in segments:
        result.append(segment['input_ids'])
        result.append(segment['attention_mask'])
        #result.append(torch.tensor(segment['input_ids']))
        #result.append(torch.tensor(segment['attention_mask']))
    return (result)


def process_batch(config, batch_data):
    print("当前进程号 : {}, 处理数据条数: {}".format(os.getpid(), len(batch_data)))
    tokenizer_path = config.tokenizer_path
    input_type = config.model_input_type

    batch_prepared_inputs = []
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_special_tokens=True)
    except:
        print("加载auto tokenizer失败，尝试加载bert tokenizer")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, add_special_tokens=True)
    if tokenizer is None:
        raise Exception("加载tokenizer失败")

    max_length = config.model_max_length
    for example in batch_data:
        cid = example.cid
        answer = example.answer
        model_inputs = prepare_input(tokenizer, max_length, input_type, example)
        batch_prepared_inputs.append({
            "cid": cid,
            "label": answer,
            "model_inputs": model_inputs
        })

    return batch_prepared_inputs


def chunks(lst, n):
    """将列表 lst 分成大小为 n 的块。"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def multi_process_data(dataset, config):
    # 将数据集划分为小批量
    max_workers = 8

    batch_size = (len(dataset) // max_workers) + 1
    batches = list(chunks(dataset, batch_size))

    process_batch_with_config = partial(process_batch, config)
    # 使用多进程处理数据
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        prepared_inputs_list = list(executor.map(process_batch_with_config, batches))

    # 将结果扁平化
    prepared_inputs_list = [item for sublist in prepared_inputs_list for item in sublist]
    return prepared_inputs_list

