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
import copy
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
    parser.add_argument("--model_name", default="bench", type=str)
    parser.add_argument("--model_path", default="1_RUN_EXP.models.BENCH_4", type=str)
    parser.add_argument("--do_train", default=True, type=lambda x: x.lower() == 'yes', required=False)
    parser.add_argument("--do_valid", default=True, type=lambda x: x.lower() == 'yes', required=False)
    parser.add_argument("--do_test", default=True, type=lambda x: x.lower() == 'yes', required=False)
    # 路径
    parser.add_argument("--train_file_path", default="", type=str)
    parser.add_argument("--dev_file_path", default="", type=str)
    parser.add_argument("--test_file_path", default="", type=str)
    parser.add_argument("--dataset_path", default="/disk2/huanggj/ACMRC_EXP_V202306/3_DATASET/ACRC/4_inputs/10_Fold", type=str)
    parser.add_argument("--pretrain_model_path", default="/disk2/huanggj/ACMRC_EXPERIMENT/pretrain/BERT", type=str)
    parser.add_argument("--output_dir", default="/disk2/huanggj/ACMRC_EXP_V202306/output", type=str)
    parser.add_argument("--result_file", default="../result/baseline_comprison/result.txt", type=str)
    # 输入
    parser.add_argument("--input_context_type", default="context", type=str)
    parser.add_argument("--input_options_type", default="options", type=str)
    parser.add_argument("--input_question_type", default="question", type=str)
    parser.add_argument("--model_input_type", default="4", type=str)
    parser.add_argument("--option_add_letter", default=False, type=lambda x: x.lower() == 'true', required=False)
    # 超参数
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--learning_rate", default=2e-6, type=float)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument("--batch_size", default=8, type=int)

    args, unknown = parser.parse_known_args()

    # 转换参数类型

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
    set_random_seed(42,False)
    # 数据加载器
    ids_generater = IdsGenerater(config)

    # 加载模型
    x = import_module(args.model_path,package=__package__)
    model = x.Model(config)
    # 多GPU  数据并行
    model = torch.nn.DataParallel(model)
    # 模型放到GPU上
    model.to(config.device)
    # 保存初始模型的参数
    initial_model_state = copy.deepcopy(model.state_dict())

    # 存储所有模型的精度
    accuracies = []
    base_dir = ''

    # 十折交叉验证
    k = 10
    for i in range(k):
        # 设置当前数据集的路径id
        ids_generater.set_k_fold_path(i)

        # 加载初始模型参数
        model.load_state_dict(copy.deepcopy(initial_model_state))


        model = trainer(config=config, model=model, ids_generater=ids_generater)


        acc, avg_loss, report, confusion = predictor(config=config, model=model, ids_generater=ids_generater)

        print(f"fold : {i} , acc : {acc}")

        # 存储精度
        accuracies.append(acc)

    # 计算平均精度
    print(accuracies)
    average_accuracy = np.mean(accuracies)

    print(f"Avg accuracy across 10-folds: {average_accuracy}")




