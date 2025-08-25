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



def NCR2ACRC(path):
    cnt = 0
    file = open(path, 'r', encoding='UTF-8')
    NCR_list = json.load(file)
    ACRC_MOD_list = []
    for d in NCR_list:
        cid = d['ID']
        context = d['Content']
        Questions = d['Questions']

        for q in Questions:
            question = q['Question']
            options = q['Choices']
            if len(options) != 4:
                cnt=cnt+1
                continue
            answer = q['Answer']
            qas_dict = dict()
            qas_dict['question'] = question
            qas_dict['options'] = options
            qas_dict['answer'] = answer
            qas = [qas_dict]
            data = dict()
            data['context'] = context
            data['qas'] = qas
            data['cid'] = cid
            ACRC_MOD_list.append(data)
    print(cnt)
    return ACRC_MOD_list



def delete_letter(options):
    new_options = []
    for option in options:
        new_option = option.replace('A.', '').replace('B.', '').replace('C.', '').replace('D.', '')
        new_option = new_option.replace('A．', '').replace('B．', '').replace('C．', '').replace('D．', '')
        new_option = new_option.replace('A、', '').replace('B、', '').replace('C、', '').replace('D、', '')
        new_options.append(new_option)
    return new_options

def write_json(final_list, path):
    dataset = {
        'version': 'ncr-data',
        'data': final_list
    }

    if os.path.exists(path):
        os.remove(path)
    json_dump(dataset, path)


if __name__ == '__main__':
    path_NCR_train = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/NCR/train_2.json'
    path_NCR_dev = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/NCR/dev_2.json'
    path_NCR_test = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/NCR/test_2.json'

    des_path_train = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/NCR/ncr_train.json'
    des_path_dev = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/NCR/ncr_dev.json'
    des_path_test = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/NCR/ncr_test.json'

    # 加载GCRC
    ncr_train = NCR2ACRC(path_NCR_train)
    ncr_dev = NCR2ACRC(path_NCR_dev)
    ncr_test = NCR2ACRC(path_NCR_test)

    write_json(ncr_train, des_path_train)
    write_json(ncr_dev, des_path_dev)
    write_json(ncr_test, des_path_test)

