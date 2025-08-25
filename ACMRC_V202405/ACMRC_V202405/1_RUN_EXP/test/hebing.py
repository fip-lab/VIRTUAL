#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : hebing.py
@Author  : huanggj
@Time    : 2023/12/19 11:32
"""
import json

# 读取第一个 JSON 文件
with open('/disk2/huanggj/ACMRC_EXP_V202306/3_DATASET/ACRC/4_inputs/acrc_train.json', 'r', encoding='utf-8') as file1:
    data1 = json.load(file1)['data']

# 读取第二个 JSON 文件
with open('/disk2/huanggj/ACMRC_EXP_V202306/3_DATASET/ACRC/4_inputs/acrc_dev.json', 'r', encoding='utf-8') as file2:
    data2 = json.load(file2)['data']

# 读取第三个 JSON 文件
with open('/disk2/huanggj/ACMRC_EXP_V202306/3_DATASET/ACRC/4_inputs/acrc_test.json', 'r', encoding='utf-8') as file3:
    data3 = json.load(file3)['data']

# 合并数据
merged_data = data1 + data2 + data3

w = {"version":"acrc-all", "data":merged_data}
# 写入合并后的数据到新的 JSON 文件
with open('/disk2/huanggj/ACMRC_EXP_V202306/3_DATASET/ACRC/4_inputs/acrc_all.json', 'w', encoding='utf-8') as merged_file:
    json.dump(merged_data, merged_file, ensure_ascii=False, indent=2)

print("合并完成，并写入到 merged_file.json 文件中。")
