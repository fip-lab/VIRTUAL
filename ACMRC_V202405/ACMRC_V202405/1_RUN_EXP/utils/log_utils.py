#encoding:utf-8
import os
import logging
from pathlib import Path
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
'''
日志模块
1. 同时将日志打印到屏幕跟文件中
2. 默认值保留近30天日志文件
'''

def get_logger(log_name):
    # 创建完整的日志路径
    log_dir = Path("/disk2/huanggj/ACMRC_EXPERIMENT") / log_name / "log"
    
    # 递归创建所有必要的父目录
    os.makedirs(log_dir, exist_ok=True)

    if log_name not in Logger.manager.loggerDict:
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)
        handler = TimedRotatingFileHandler(filename=str(log_dir / f"{log_name}.log"), when='D', backupCount=30)
        datefmt = '%Y-%m-%d %H:%M:%S'
        format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
        formatter = logging.Formatter(format_str, datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        handler = TimedRotatingFileHandler(filename=str(log_dir / "ERROR.log"), when='D', backupCount=30)
        datefmt = '%Y-%m-%d %H:%M:%S'
        format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
        formatter = logging.Formatter(format_str, datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
    logger = logging.getLogger(log_name)
    return logger
