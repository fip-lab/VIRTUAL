#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import time
from datetime import timedelta
"""
@File    : time_util.py
@Author  : huanggj
@Time    : 2022/12/3 14:29
"""
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif))).seconds


