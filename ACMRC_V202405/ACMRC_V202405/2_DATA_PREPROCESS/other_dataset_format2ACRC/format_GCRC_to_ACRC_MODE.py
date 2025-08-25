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



def GCRC2ACRC(path, cid):
    file = open(path, 'r', encoding='UTF-8')
    total_list = json.load(file)['data']

    final_list = []
    for d in total_list:
        context = d['passage']
        question = d['question']
        options = d['options']
        answer = d['answer']
        data = dict()
        data['context'] = context
        data['context_type'] = "[现代文]"
        qas = dict()
        qas['question'] = "[现代文]" + question
        qas['options'] = options
        qas['answer'] = answer
        data['qas'] = [qas]
        data['cid'] = cid
        cid = cid + 1
        final_list.append(data)
    return final_list

def ACRCFormat(path):
    file = open(path, 'r', encoding='UTF-8')
    total_list = json.load(file)['data']

    final_list = []
    for d in total_list:
        d['context_type'] = "[古文]"
        d['qas'][0]['question'] = "[古文]" + d['qas'][0]['question']
        final_list.append(d)

    return final_list

def write_json(final_list, path):
    dataset = {
        'version': 'prompt-train',
        'data': final_list
    }

    if os.path.exists(path):
        os.remove(path)
    json_dump(dataset, path)


if __name__ == '__main__':
    path_gcrc_train = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/GCRC/train.json'
    path_gcrc_dev = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/GCRC/valid.json'

    path_acrc_train = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/train.json'
    des_path = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC_GCRC_PROMPT/train.json'

    path_acrc_dev = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/dev.json'
    des_path_dev = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC_GCRC_PROMPT/dev.json'

    path_acrc_test = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/test.json'
    des_path_test = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC_GCRC_PROMPT/test.json'

    # 加载GCRC
    gcrc_train = GCRC2ACRC(path_gcrc_train, 10000)
    gcrc_dev = GCRC2ACRC(path_gcrc_dev, 20000)

    # 加载ACRC
    acrc_train = ACRCFormat(path_acrc_train)
    acrc_dev = ACRCFormat(path_acrc_dev)
    acrc_test = ACRCFormat(path_acrc_test)

    # 合并数据
    gcrc_train.extend(acrc_train)
    gcrc_train.extend(gcrc_dev)
    random.shuffle(gcrc_train)
    print("###")


    write_json(gcrc_train, des_path)
    write_json(acrc_dev, des_path_dev)
    write_json(acrc_test, des_path_test)


