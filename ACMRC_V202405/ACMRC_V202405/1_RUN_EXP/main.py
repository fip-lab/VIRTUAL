#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : main.py
@Author  : huanggj
@Time    : 2023/2/16 23:32
"""
import argparse
import config
import os
import random
import numpy as np
import torch.nn.parallel
from train import trainer
from predict import predictor
from importlib import import_module
from data_processor import IdsGenerater

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", default="default_task", type=str)
    parser.add_argument("--model_name", default="cmp_t5", type=str)
    parser.add_argument("--model_path", default="models_compare.cmp_t5", type=str)
    parser.add_argument("--do_train", default=True, type=lambda x: x.lower() == 'yes', required=False)
    parser.add_argument("--do_valid", default=True, type=lambda x: x.lower() == 'yes', required=False)
    parser.add_argument("--do_test", default=True, type=lambda x: x.lower() == 'yes', required=False)
    # 路径
    parser.add_argument("--train_file_path", default="../../../data/train_data.json", type=str)
    parser.add_argument("--dev_file_path", default="../../../data/val_data.json", type=str)
    parser.add_argument("--test_file_path", default="../../../data/test_data.json", type=str)
    parser.add_argument("--dataset_path", default="../3_DATASET/ACRC/4_inputs/", type=str)
    parser.add_argument("--pretrain_model_path", default="./pretrain/mt5-base", type=str)
    parser.add_argument("--output_dir", default="./checkpoint", type=str)
    parser.add_argument("--result_file", default="../4_RESULT/baseline_comprison/result.txt", type=str)
    # 输入
    parser.add_argument("--input_context_type", default="context", type=str)
    parser.add_argument("--input_options_type", default="options", type=str)
    parser.add_argument("--input_question_type", default="question", type=str)
    parser.add_argument("--model_input_type", default="1", type=str)
    parser.add_argument("--option_add_letter", default=False, type=lambda x: x.lower() == 'true', required=False)
    # 超参数
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--learning_rate", default=2e-6, type=float)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int)

    args, unknown = parser.parse_known_args()
    return args


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 设置GPU可见性
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # 解析参数
    args = arg_parse()
    config = config.Config(args)

    # 设置随机数种子，保证运行结果一致
    set_random_seed(args.seed, True)

    # 数据加载器
    ids_generater = IdsGenerater(config)

    # 加载模型
    x = import_module(args.model_path)
    model = x.Model(config)
    # 多GPU  数据并行
    model = torch.nn.DataParallel(model)
    # 模型放到GPU上
    model.to(config.device)

    if args.do_train:
        trainer(config=config, model=model, ids_generater=ids_generater)

    if args.do_test:
        predictor(config=config, model=model, ids_generater=ids_generater)




