#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : extractor_ner.py
@Author  : huanggj
@Time    : 2022/8/4 14:31
"""
from transformers import BertTokenizer, RobertaForTokenClassification
from extract.extractor_base import ExtractionBase
from transformers import BertTokenizerFast, AutoModelForTokenClassification

#model_name = 'ethanyt/guwen-ner'111

class ExtractorNER(ExtractionBase):

    # 初始化方法
    def __init__(self,option_model_name, passage_model_name):
        ExtractionBase.__init__(self,option_model_name, passage_model_name)
        print("option model : <%s>"%(option_model_name))
        print("passage model : <%s>"%(passage_model_name))
        if option_model_name == 'ethanyt/guwen-ner':
            self.option_tokenizer = BertTokenizer.from_pretrained(self.option_model_name)
            self.option_model = RobertaForTokenClassification.from_pretrained(self.option_model_name)

        if option_model_name == 'ckiplab/bert-base-chinese-ner':
            self.option_tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
            self.option_model = AutoModelForTokenClassification.from_pretrained(self.option_model_name)

        if passage_model_name == 'ethanyt/guwen-ner':
            self.passage_tokenizer = BertTokenizer.from_pretrained(self.passage_model_name)
            self.passage_model = RobertaForTokenClassification.from_pretrained(self.passage_model_name)

        if passage_model_name == 'ckiplab/bert-base-chinese-ner':
            self.passage_tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
            self.passage_model = AutoModelForTokenClassification.from_pretrained(self.passage_model_name)

    # 抽取入口:
    def extract(self, option_list, passage):
        # 参数检查
        if option_list is None or len(option_list) != 4:
            print(option_list)
            print('ERROR *** 选项个数错误 ***请检查')
            raise Exception

        if passage is None :
            print('ERROR *** 文章为None ***请检查')
            raise Exception

        # for op in option_list:
        #     op = op.replace('A','').replace('B','').replace('C','').replace('D','')

        # 选项ner
        print("option A")
        ner_result_option_A = self.ner_single(option_list[0].replace('A',''),self.OPTION)
        print("option B")
        ner_result_option_B = self.ner_single(option_list[1].replace('B',''),self.OPTION)
        print("option C")
        ner_result_option_C = self.ner_single(option_list[2].replace('C',''),self.OPTION)
        print("option D")
        ner_result_option_D = self.ner_single(option_list[3].replace('D', ''),self.OPTION)
        print("all options")
        ner_result_option_ALL = self.ner_single(option_list[0].replace('A','') + option_list[1].replace('B','') + option_list[2].replace('C','') + option_list[3].replace('D', ''),self.OPTION)

        # 文章ner
        #passage_ner_result = self.ner_batch(passage,self.PASSAGE)
        passage_ner_result = None
        # 抽取每个选项相近的句子
        extracte_sentences_option_A = self.extract_similar_sentence_of_option(ner_result_option_A,passage_ner_result)
        extracte_sentences_option_B = self.extract_similar_sentence_of_option(ner_result_option_B,passage_ner_result)
        extracte_sentences_option_C = self.extract_similar_sentence_of_option(ner_result_option_C,passage_ner_result)
        extracte_sentences_option_D = self.extract_similar_sentence_of_option(ner_result_option_D,passage_ner_result)

        return ner_result_option_ALL

    # 单个 Named Entity Recognition
    def ner_single(self, sentence, ner_kind):
        logits = None
        entity_list = None
        # NER option
        if ner_kind == self.OPTION:
            if self.option_model_name == 'ethanyt/guwen-ner':
                tokens = self.option_tokenizer(sentence, return_tensors='pt')
                logits = self.option_model(**tokens).logits
                logit_result = logits.argmax(axis=2).tolist()
                # 可视化   logit_result[0][1:]  1:  第一个是special token
                self.visualization_guwen_ner(logit_result[0][1:], sentence)
                entity_list = self.get_ner_result_guwen_ner(logit_result[0][1:], sentence)

            if self.option_model_name == 'ckiplab/bert-base-chinese-ner':
                tokens = self.option_tokenizer(sentence, return_tensors='pt')
                logits = self.option_model(**tokens).logits
                logit_result = logits.argmax(axis=2).tolist()
                self.visualization_chinese_ner(logits, logit_result, logit_result[0][1:], sentence)
                entity_list = self.get_ner_result_chinese_ner(logits, logit_result, logit_result[0][1:], sentence)

        # ner passage
        if ner_kind == self.PASSAGE:
            if self.passage_model_name == 'ethanyt/guwen-ner':
                tokens = self.passage_tokenizer(sentence, return_tensors='pt')
                logits = self.passage_model(**tokens).logits
                logit_result = logits.argmax(axis=2).tolist()
                # 可视化   logit_result[0][1:]  1:  第一个是special token
                self.visualization_guwen_ner(logit_result[0][1:], sentence)
                entity_list = self.get_ner_result_guwen_ner(logit_result[0][1:], sentence)

            if self.passage_model_name == 'ckiplab/bert-base-chinese-ner':
                tokens = self.passage_tokenizer(sentence, return_tensors='pt')
                logits = self.passage_model(**tokens).logits
                logit_result = logits.argmax(axis=2).tolist()
                self.visualization_chinese_ner(logits, logit_result, logit_result[0][1:], sentence)
                entity_list = self.get_ner_result_chinese_ner(logits, sentence)

        entity_list = set(entity_list)
        print(entity_list)
        print('***********************************')
        return entity_list

    # 批量 Named Entity Recognition
    def ner_batch(self, passage_or_sentences,ner_kind):
        total_ner_list = []
        # 以句号分割句子
        sentence_list = passage_or_sentences.split('。')
        # 循环抽取
        index = 1
        for sentence in sentence_list:
            print("sentence %d"%(index))
            total_ner_list.append(self.ner_single(sentence,ner_kind))
            index = index + 1
        return total_ner_list

    # 抽取每个选项相近的句子
    def extract_similar_sentence_of_option(self, ner_result_option, passage_ner_result):
        pass

    def visualization_chinese_ner(self, logits,logit_result, tags, text):
        # print(logits.shape)
        # print(logit_result)
        # print(tags)
        # print('tags length %d '%(len(tags) + 1))
        # print(text)
        # print('text length %d ' % (len(text)))

        in_span = False
        total_span_list = []
        current_span_list = []
        result = []
        for i, (tag, token) in enumerate(zip(tags, text + " ")):
            if tag == 0  and in_span:
                result.append("」")
                in_span = False

            if tag != 0 and not in_span:
                result.append("「")
                in_span = True

            result.append(token)
        print(tags)
        print('ner:' + ''.join(result))
        return total_span_list


    def get_ner_result_chinese_ner(self, logits,logit_result, tags, text):
        in_span = False
        total_span_list = []
        current_span_list = []
        result = []
        for i, (tag, token) in enumerate(zip(tags, text + " ")):
            if tag == 0 and not in_span:
                continue

            if tag == 0 and in_span:
                if len(current_span_list) > 0:
                    total_span_list.append(''.join(current_span_list))
                    current_span_list = []
                in_span = False
                continue

            if tag != 0 and not in_span:
                in_span = True

            current_span_list.append(token)
        return total_span_list

    # 打印ner的情况，并返回所有实体: list
    def visualization_guwen_ner(self, tags, text):
        in_span = False
        total_span_list = []
        current_span_list = []
        result = []
        for i, (tag, token) in enumerate(zip(tags, text + " ")):
            if tag in (0, 1) and in_span:
                result.append("」")
                in_span = False

            if tag == 3 and in_span:
                result.append("」")
                if len(current_span_list) > 0:
                    # print(''.join(current_span_list))
                    total_span_list.append(''.join(current_span_list))
                    current_span_list = []
                in_span = False

            if tag == 3 and not in_span:
                result.append("「")
                if len(current_span_list) > 0:
                    # print(''.join(current_span_list))
                    total_span_list.append(''.join(current_span_list))
                    current_span_list = []
                in_span = True

            if in_span:
                current_span_list.append(token)

            result.append(token)
        print(tags)
        print('ner:' + ''.join(result))
        return total_span_list

    # 打印ner的情况，并返回所有实体: list
    def get_ner_result_guwen_ner(self, tags, text):
        in_span = False
        total_span_list = []
        current_span_list = []
        for i, (tag, token) in enumerate(zip(tags, text + " ")):
            if tag in (0, 1, 2) and not in_span:
                continue

            if tag == 4 and in_span:
                current_span_list.append(token)
                continue

            if tag in (0, 1, 2, 3) and in_span:
                if len(current_span_list) > 0:
                    total_span_list.append(''.join(current_span_list))
                    current_span_list = []
                in_span = False

                if tag == 3:
                    in_span = True
                    current_span_list.append(token)
                continue

            if tag == 3 and not in_span:
                in_span = True
                current_span_list.append(token)

        return total_span_list




extractor = ExtractorNER("ckiplab/bert-base-chinese-ner",'ethanyt/guwen-ner')
f = open('../dataset/csv/train.csv')
lines = f.readlines()
index = 1
NOT_ENTITY_NUM = 0
for line in lines:
    arr = line.split(',')
    q_a= arr[0]
    a_list = q_a.split('[SEP]')[1:]
    passage = arr[1]
    try:
        ner_list = extractor.extract(a_list, passage)
    except:
        print("ERROR %d "%index)
        print(index)
        index = index + 1

        continue
    print("res=================================================================")
    print(ner_list)
    if len(ner_list) == 0:
        print("NO ENTITY !!! line %d "%index)
        NOT_ENTITY_NUM = NOT_ENTITY_NUM + 1
    print(index)
    index = index + 1

#print(NOT_ENTITY_NUM)
#print(a_list)
#print(passage)
#extractor = ExtractionNER('ethanyt/guwen-ner', 'ethanyt/guwen-ner')

#extractor = ExtractionNER("ckiplab/bert-base-chinese-ner",'ckiplab/bert-base-chinese-ner')


