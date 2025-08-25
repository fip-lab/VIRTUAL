#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : test_excel.py
@Author  : huanggj
@Time    : 2023/6/15 11:22
"""
import pandas as pd

def read_excel_params(file_path, sheet_name=0):
    # 使用pandas读取excel文件
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=1, engine='openpyxl')
    # 将DataFrame转化为字典列表，每一行是一个任务，列名是参数名，列值是参数值
    tasks = df.to_dict(orient='records')
    # 把 Nan值转换为None
    for i in range(len(tasks)):
        for key, value in tasks[i].items():
            if pd.isnull(value):
                tasks[i][key] = None
    return tasks

