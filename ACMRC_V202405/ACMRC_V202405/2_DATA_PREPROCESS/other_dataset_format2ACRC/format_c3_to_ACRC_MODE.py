#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : format_c3_to_ACRC_MODE.py
@Author  : huanggj
@Time    : 2023/2/14 17:02
"""

import io, json, codecs, os
import random
from tqdm import tqdm
ave = lambda x : sum(x)/len(x)
codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)
json_dumpsl = lambda d: json.dumps(d, ensure_ascii=False)

path = '/disk2/huanggj/ACMRC_INTEGRATION/data_c3/c3-m-dev.json'
file = open(path, 'r', encoding='UTF-8')
total_list = json.load(file)

lable2id = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
        }


final_list = []
cid = 0
s = set()
for d in total_list:
    context = d[0][0]
    for qas in d[1]:
        question = qas['question']
        choice = qas['choice']
        answer = qas['answer']

        data = dict()
        data['context'] = context
        qas_dict = dict()
        qas_dict['question'] = question

        qas_dict['answer'] = lable2id[choice.index(answer)]
        s.add(len(choice))
        if len(choice) == 2:
            answer_idx = choice.index(answer)
            if answer_idx == 0:
                choice.append(choice[1])
                choice.append(choice[1])

            if answer_idx == 1:
                choice.append(choice[0])
                choice.append(choice[0])

        if len(choice) == 3:
            answer_idx = choice.index(answer)
            if answer_idx == 0:
                choice.append(choice[1])

            if answer_idx == 1:
                choice.append(choice[0])

            if answer_idx == 2:
                choice.append(choice[0])
        qas_dict['options'] = choice
        qas_dict['qid'] = 0
        data['qas'] = [qas_dict]
        data['cid'] = cid
        cid = cid + 1
        final_list.append(data)

print(s)

dataset = {
        'version': 'acmrc-test',
        'data': final_list
}

train_path = '/disk2/huanggj/ACMRC_INTEGRATION/data_c3/dev.json'

if os.path.exists(train_path):
    os.remove(train_path)



json_dump(dataset, train_path)