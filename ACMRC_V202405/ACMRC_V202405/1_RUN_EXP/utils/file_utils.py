#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : file_utils.py
@Author  : huanggj
@Time    : 2023/2/17 9:31
"""
import six
import datetime
import os  # 导入负责处理操作系统相关事务的os模块

def is_empty_dir(path):
    if len(os.listdir(path)) == 0:
        return True
    return False

def convert_to_unicode(text):
    # 如果当前是python3环境
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    # 如果当前是python2环境
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def get_latest_excel(path):
    if not os.path.isdir(path):
        raise Exception("{} 不是一个文件夹".format(path))
    # file_list = os.listdir(path)
    # latest_file = ''
    # latest_time = None
    # for file_name in file_list:
    #     if not file_name.endswith('.xlsx') and not file_name.endswith('.xls'):
    #         continue
    #     file = path + '/' + file_name
    #     if not os.path.isfile(file):
    #         continue
    #     modify_time = os.path.getmtime(file)
    #     if latest_time == None:
    #         latest_time = modify_time
    #         latest_file = file
    #     elif modify_time > latest_time:
    #         latest_time = modify_time
    #         latest_file = file
    #     else:
    #         continue
    latest_file = None
    latest_time = datetime.datetime.fromtimestamp(0)
    for file in os.listdir(path):
        if file.endswith('.xlsx') or file.endswith('.xls'):
            file_path = os.path.join(path, file)
            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            if mod_time > latest_time:
                latest_time = mod_time
                latest_file = file_path

    if latest_file == '':
        raise  Exception("{}, 找不到excel文件".format(path))
    print("##  lastest file ## : {}".format(latest_file))
    return latest_file



