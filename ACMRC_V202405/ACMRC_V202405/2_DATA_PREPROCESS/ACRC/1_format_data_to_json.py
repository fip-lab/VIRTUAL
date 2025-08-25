#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : 1_format_data_to_json.py  数据集处理类重构
@Author  : huanggj
@Time    : 2022/10/29 15:20
"""
import os
import json
import re
import random

class DataFormatter:
    def __init__(self, input_dir, output_dir, filter_file):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.filter_file = filter_file
        # output dir
        self.json_output_dir = self.output_dir
        self.answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        # 加载过滤词
        self.load_filter_words()

    def format(self):
        data_list = self.load_original_data()
        data_list = self.data_filter(data_list)
        # 数据混洗
        random.shuffle(data_list)
        # 输出格式化后的数据
        self.format2json(data_list)




    def format2json(self, data_list):
        '''
        格式化数据为json集成格式    文章 - 问题
        @return:
        '''
        format_list = []
        cid = 0
        for data_tuple in data_list:
            j = dict()
            passage = data_tuple[0]
            q_a_arr = data_tuple[1]
            question = q_a_arr[0]
            option_list = q_a_arr[1:]
            answer = data_tuple[2]
            json_dict = dict()
            json_dict['question'] = question
            json_dict['answer'] = answer
            json_dict['options'] = option_list

            j['context'] = passage

            # 题目类型  正确 - 1 / 不正确 - 0
            question_type = 1
            if '有误' in question or '错误' in question or '不正确' in question or '不对的' in question or '不恰切' in question:
                question_type = 0

            if question_type == 1:
                correctness = [0, 0, 0, 0]
                correctness[self.answer_map.get(answer)] = 1
                json_dict['correctness'] = correctness
            if question_type == 0:
                correctness = [1, 1, 1, 1]
                correctness[self.answer_map.get(answer)] = 0
                json_dict['correctness'] = correctness

            json_dict['question_type'] = question_type

            json_dict['qid'] = 1
            j['qas'] = [json_dict]
            j['cid'] = cid
            cid = cid + 1
            format_list.append(j)

        print("数据集个数: %d"%len(format_list))

        data_dict = dict()
        data_dict['version'] = "acmrc-data"
        data_dict['data'] = format_list

        # if not os.path.exists(self.json_output_dir):
        #     os.mkdir(self.json_output_dir)

        with open(self.json_output_dir, 'w',encoding='utf8') as f:
            json.dump(data_dict, f,ensure_ascii=False,indent = 3,sort_keys=False)

    # 数据过滤
    def data_filter(self, data_list):
        filtered_data = []
        s = set()
        f_s = set()
        for json_dict in data_list:
            for key in json_dict.keys():
                data_dict = json_dict.get(key)

                passage = data_dict['context']
                # 文章为空
                if passage == '':
                    continue

                # if "①" in passage:
                #     print(passage)
                #     continue



                passage = passage.replace('.', '')
                passage = passage.replace(' ', '')
                passage = passage.replace('\t', '')
                passage = passage.replace(' ', '')
                passage_sentence_list = re.findall('[^。]*。', passage)

                p_length = len(passage_sentence_list)
                # 去掉文章小于12句的
                if p_length < 12:
                    continue

                passage_new = ''
                for sentence in passage_sentence_list:
                    passage_new = passage_new + sentence.strip()
                passage = passage_new
                question_dict = data_dict['question-answer']
                for q_a in question_dict.keys():
                    answer = question_dict.get(q_a)
                    # 答案不是 ABCD其中之一
                    if answer not in ('A', 'B', 'C', 'D'):
                        continue
                    # question_answer中没有 ABCD
                    if 'A' not in q_a or 'B' not in q_a or 'C' not in q_a or 'D' not in q_a:
                        continue
                    # 过滤题型
                    if self.filter(q_a):
                        # 替换特殊符号
                        q_a = q_a.replace(' ', '')
                        q_a = q_a.replace('\t', '')
                        q_a = q_a.replace(' ', '')
                        q_a = q_a.replace(' ', '')
                        q_a = q_a.replace('A．', '[SEP]').replace('B．', '[SEP]').replace('C．', '[SEP]').replace('D．','[SEP]')
                        q_a = q_a.replace('A.', '[SEP]').replace('B.', '[SEP]').replace('C.', '[SEP]').replace('D.','[SEP]')
                        q_a = q_a.replace('A,', '[SEP]').replace('B,', '[SEP]').replace('C,', '[SEP]').replace('D,','[SEP]')
                        q_a = q_a.replace('A，', '[SEP]').replace('B，', '[SEP]').replace('C，', '[SEP]').replace('D，','[SEP]')
                        q_a = q_a.replace('A、', '[SEP]').replace('B、', '[SEP]').replace('C、', '[SEP]').replace('D、','[SEP]')
                        q_a = q_a.replace('A：', '[SEP]').replace('B：', '[SEP]').replace('C：', '[SEP]').replace('D：','[SEP]')
                        q_a = q_a.replace('A', '[SEP]').replace('B', '[SEP]').replace('C', '[SEP]').replace('D','[SEP]')
                        q_a = q_a.replace(',', '，')
                        q_a_arr = q_a.split("[SEP]")
                        if len(q_a_arr) != 5:
                            continue
                        for i in range(4):
                            option = q_a_arr[i + 1]
                            if not option.endswith('。'):
                                q_a_arr[i + 1] = option + '。'
                            if option.endswith('.'):
                                q_a_arr[i + 1] = option[:len(option)-2] + '。'

                        filtered_data.append((passage, q_a_arr, answer))
        # 来源网址
        #         s.add(key)
        #         #a = re.compile(r'[a-zA-Z]+://[^\s]*[.com|.cn]')
        #
        #         for wz in s:
        #             host_match = re.match(r"\w{4,5}://(\w|\.|:)+", wz)
        #             # HOST = host_match and host_match.group()
        #             if host_match is not None:
        #                 HOST = host_match.group()
        #                 f_s.add(HOST)
        #             #print(HOST)
        #
        #         print("#######################")
        # for i in f_s:
        #     print(i)

        return filtered_data

    # 加载过滤词文件
    def load_filter_words(self):
        f = open(self.filter_file, encoding='utf8')
        self.filter_words = f.read().splitlines()

    # 过滤
    def filter(self, question_options):
        for filter_word in self.filter_words:
            if filter_word in question_options:
                return False
        return True

    # 加载原始数据
    def load_original_data(self):
        file_list = os.listdir(self.input_dir)
        data_list = []
        for file_name in file_list:
            if 'ancient' in file_name:
                continue

            if '10' not in file_name:
                continue

            print(file_name)
            json_file = open(self.input_dir + '/' + file_name, 'r', encoding='utf-8')
            data = json.load(json_file)
            data_list.append(data)
        return data_list

# 将一个list尽量均分成n份，限制len(list)==n，份数大于原list内元素个数则分配空list[]
def divideIntoNstrand(listTemp, n):
    twoList = [ [] for i in range(n)]
    for i,e in enumerate(listTemp):
        twoList[i%n].append(e)
    return twoList

if __name__ == '__main__':
    # 源数据目录
    input_dir = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/original_data'
    # 输出数据目录
    output_dir = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/temp_data/acrc_1.json'
    # 过滤关键字的文件路径
    filter_file = '/disk2/huanggj/ACMRC_EXPERIMENT/data_preprocess/ACRC/rc_filter.txt'
    # 数据格式化对象
    data_formatter = DataFormatter(input_dir, output_dir, filter_file)
    # 输出数据
    data_formatter.format()

