#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : format_vgaokao_to_ACRC_MODE.py
@Author  : huanggj
@Time    : 2023/2/14 17:01
"""
#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : format_c3_to_ACRC_MODE.py
@Author  : huanggj
@Time    : 2023/2/14 17:02
"""

import io, json, codecs, os, copy
import random
from tqdm import tqdm
ave = lambda x : sum(x)/len(x)
codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

#json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)
json_dumpsl = lambda d: json.dumps(d, ensure_ascii=False)

path = '/disk2/huanggj/ACMRC_INTEGRATION/data_vgaokao/train.json'
file = open(path, 'r', encoding='UTF-8')
total_list = json.load(file)['data']

lable2id = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
        }


final_list = []
cid = 0
for d in total_list:
    context = d['context']
    for qas in d['qas']:
        data = dict()
        data['context'] = context
        data['qas'] = [qas]
        qas_dict = dict()
        data['cid'] = cid
        cid = cid + 1
        final_list.append(data)

dataset = {
        'version': 'vgk-train',
        'data': final_list
}

train_path = '/disk2/huanggj/ACMRC_INTEGRATION/data_vgaokao/train_vgk.json'

if os.path.exists(train_path):
    os.remove(train_path)



json_dump(dataset, train_path)