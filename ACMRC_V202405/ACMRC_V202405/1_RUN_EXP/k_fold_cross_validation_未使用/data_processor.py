#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : data_processor.py
@Author  : huanggj
@Time    : 2023/2/16 23:41
"""
# encoding=utf-8
import torch
import json
from tqdm import tqdm
from utils.file_utils import convert_to_unicode
from data_loader_multiprocess import multi_process_data




class InputExample(object):
    def __init__(self, question, options, passage, options_evidence, answer, cid, question_type):
        self.question = question
        self.question_type = question_type
        self.options = options
        self.passage = passage
        self.options_evidence = options_evidence
        self.answer = answer
        self.cid = cid

class Processor():
    def __init__(self, context_type, question_type, options_type, option_add_letter, id2label, label2id):
        self.context_type = context_type
        self.question_type = question_type
        self.options_type = options_type
        self.option_add_letter = option_add_letter
        self.id2label = id2label
        self.label2id = label2id

    @classmethod
    def _read_json(self, input_file):
        """Reads a comma separated value file."""
        with open(input_file, 'r') as load_f:
            load_data = json.load(load_f)
            return load_data

    # 获取json格式的数据
    def get_examples_json(self, file_path):
        return self._create_examples_json(self._read_json(file_path))

    # 获取一个InputExample List
    def _create_examples_json(self, json_data):
        data = json_data



        # 判断有没有包装json
        if not isinstance(data, list):
            data = data['data']


        examples = []
        # 遍历包装数据
        for (i, json_obj) in enumerate(data):
            cid = json_obj['cid']
            qas = json_obj['qas']
            passage = convert_to_unicode(json_obj[self.context_type])
            for (j, qas_obj) in enumerate(qas):
                question = convert_to_unicode(qas_obj[self.question_type])
                options = qas_obj[self.options_type]
                label = self.label2id[qas_obj['answer']]
                question_type = qas_obj['question_type']
                # 判断有没有options_evidence
                if 'options_evidence' in qas_obj.keys():
                    options_evidence = qas_obj['options_evidence']
                else:
                    options_evidence = []

                for index in range(4):
                    options[index] = convert_to_unicode(options[index])
                    # 加不加字母
                    if self.option_add_letter:
                        options[index] = self.id2label[index] + options[index]
                # 组装成一个InputExample对象，添加到一个list中
                examples.append(InputExample(question=question,
                                             options=options,
                                             passage=passage,
                                             options_evidence=options_evidence,
                                             answer=label,
                                             cid=cid,
                                             question_type=question_type))
        return examples


class IdsGenerater(object):

    def __init__(self, config):
        """

        @param data_dir:  数据集目录
        @param intput_context_type:  预训练模型目录
        @param intput_options_type: 问题最大长度
        @param intput_question_type:  文章片段1最大长度
        @param passage_two_max_seq_length:  文章片段2最大长度
        """
        self.config = config
        self.logger = config.logger
        self.device = config.device
        self.dataset_path = config.dataset_path
        self.train_file_path = config.train_file_path
        self.dev_file_path = config.dev_file_path
        self.test_file_path = config.test_file_path
        # 组装数组的选项
        self.processor = Processor(config.input_context_type,
                                   config.input_question_type,
                                   config.input_options_type,
                                   config.option_add_letter,
                                   config.id2label,
                                   config.label2id)

        self.label2id = config.label2id
        self.id2label = config.id2label
        self.tokenizer_path = config.tokenizer_path

        self.k_fold_idx = -1

        # 模型的输入类型， 决定怎么组装数据
        self.model_input_type = config.model_input_type

    # 获取训练数据
    def get_train_ids(self):
        train_file_path = self.train_file_path
        if self.k_fold_idx != -1:
            train_file_path = self.dataset_path + '/' + 'fold_' + str(self.k_fold_idx)  + '/train.json'
        train_examples = self.processor.get_examples_json(train_file_path)
        # 多线程进行tokenizer
        self.logger.info("加载训练集数据")
        train_features = multi_process_data(train_examples, self.config)
        return train_features

    # 获取验证数据
    def get_dev_ids(self):
        dev_file_path = self.dev_file_path
        if self.k_fold_idx != -1:
            dev_file_path = self.dataset_path + '/'  + 'fold_' + str(self.k_fold_idx)  + '/dev.json'
        dev_examples = self.processor.get_examples_json(dev_file_path)
        # 多线程进行tokenizer
        self.logger.info("加载验证集数据")
        dev_features = multi_process_data(dev_examples, self.config)
        return dev_features

    #  获取测试数据
    def get_test_ids(self):
        test_file_path = self.test_file_path
        if self.k_fold_idx != -1:
            test_file_path = self.dataset_path + '/' + 'fold_' + str(self.k_fold_idx)  + '/test.json'
        test_examples = self.processor.get_examples_json(test_file_path)
        # 多线程进行tokenizer
        self.logger.info("加载测试集数据")
        test_features = multi_process_data(test_examples, self.config)
        return test_features

    def get_all_ids(self):
        train_examples = self.processor.get_examples_json(self.dataset_path + '/' + 'fold_' + str(self.k_fold_idx) + '/train.json')
        dev_examples = self.processor.get_examples_json(self.dataset_path + '/'  + 'fold_' + str(self.k_fold_idx) + '/dev.json')
        test_examples = self.processor.get_examples_json(self.dataset_path + '/'  + 'fold_' + str(self.k_fold_idx) + '/test.json')
        all_examples = train_examples + dev_examples + test_examples
        # 多线程进行tokenizer
        all_features = multi_process_data(all_examples, self.config)
        return all_features

    def set_k_fold_path(self, k_fold_idx):
        self.k_fold_idx = k_fold_idx