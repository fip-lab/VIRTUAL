#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : config.py
@Author  : huanggj
@Time    : 2023/2/17 1:16
"""
import torch, os
import  datetime
from utils.log_utils import get_logger
from utils.random_utils import generate_random_str

class Config(object):
    MODEL_INPUT_TYPE1 = "1"
    MODEL_INPUT_TYPE2_1 = "2_1"
    MODEL_INPUT_TYPE2_2 = "2_2"
    MODEL_INPUT_TYPE3_1 = "3_1"
    MODEL_INPUT_TYPE3_2 = "3_2"
    MODEL_INPUT_TYPE5 = "5"
    """
        model_input_type1 : 512（question + options + passage_1）; 

        model_input_type2_1 : 512（question + options + passage_1）+ 512（passage）;

        model_input_type2_2 : 512（question + options）+ 512（passage_1）; 
        
        model_input_type3_1 : 512（question + options）+ 
                              512（passage_1）+ 
                              512（passage_2）;
        
        model_input_type3_2 : 512（question + options + passage_1）+ 
                            512（question + options + passage_2）+ 
                            512（question + options + passage_3）;

        model_input_type5 : 512（question + options + passage_1）+ 
                            512（question + options + passage_2）+ 
                            512（question + options + passage_3）+ 
                            512（question + options + passage_4）+ 
                            512（question + options + passage_5）;
    """
    """配置参数"""
    def __init__(self, args):
        self.logger = get_logger(args.model_name)
        self.logger.info("init config")
        self.task_id = args.task_id
        self.result_file = args.result_file
        self.model_name = args.model_name
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 训练集
        self.train_file_path = args.train_file_path
        # 验证集
        self.dev_file_path = args.dev_file_path
        # 测试集
        self.test_file_path = args.test_file_path
        # 数据集目录：用于10折交叉验证
        self.dataset_path = args.dataset_path

        self.do_train = args.do_train
        self.do_valid = args.do_valid
        self.do_test = args.do_test

        # 预训练模型
        self.pretrain_model_path = args.pretrain_model_path
        self.tokenizer_path = args.pretrain_model_path

        # label
        self.label_list = ['A', 'B', 'C', 'D']
        self.label2id = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3}
        self.id2label = { 0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D'}
        self.num_label = 4
        # input type
        self.model_input_type = args.model_input_type
        self.model_path = args.model_path
        self.input_context_type = args.input_context_type
        self.input_question_type = args.input_question_type
        self.input_options_type = args.input_options_type

        # 选项加入ABCD
        self.option_add_letter = args.option_add_letter

        # funtional words list
        self.funtional_words = ["而", "何", "乎", "乃", "其", "且", "若", "为", "焉", "也", "以", "因", "于", "与", "则", "者", "之"]

        # 模型训练结果
        self.output_dir = args.output_dir
        #self.save_path =args.output_dir + "/" + self.model_name + '_' +generate_random_str(10) + '.ckpt'
        self.save_path =args.output_dir + "/" + self.model_name + '_' + str(self.task_id) + '_' + datetime.datetime.now().strftime('%m%d%H%M') + '.ckpt'

        # 若超过1000batch效果还没提升，则提前结束训练
        self.require_improvement = 1000
        # epoch
        self.epochs = args.epochs
        # batch
        self.batch_size = args.batch_size
        # 学习率
        self.learning_rate = args.learning_rate

        """
        model params
        """
        self.cnn_channels = 500
        self.kernel_size = 10
        self.stride = 1
        self.nhead = 5
        self.dropout_rate = args.dropout_rate

        self.get_model_max_length()

        self.logger.info("*** task id              *** : {}".format(self.task_id))
        self.logger.info("*** device               *** : {}".format(self.device))
        self.logger.info("*** pretrain model       *** : {}".format(self.pretrain_model_path))
        self.logger.info("*** train path           *** : {}".format(self.train_file_path))
        self.logger.info("*** dev path             *** : {}".format(self.dev_file_path))
        self.logger.info("*** test path            *** : {}".format(self.test_file_path))
        self.logger.info("*** checkpoint path      *** : {}".format(self.save_path))
        self.logger.info("*** intput_context_type  *** : {}".format(self.input_context_type))
        self.logger.info("*** intput_question_type *** : {}".format(self.input_question_type))
        self.logger.info("*** intput_options_type  *** : {}".format(self.input_options_type))
        self.logger.info("*** add options letter   *** : {}".format(self.option_add_letter))
        self.logger.info("*** model path           *** : {}".format(self.model_path))
        self.logger.info("*** model_input_type     *** : {}".format(self.model_input_type))
        self.logger.info("*** epoch                *** : {}".format(self.epochs))
        self.logger.info("*** batch_size           *** : {}".format(self.batch_size))
        self.logger.info("*** learning rate        *** : {}".format(self.learning_rate))
        self.logger.info("*** PLM length           *** : {}".format(self.model_max_length))
        self.logger.info("*** model input length   *** : {}".format(self.length))
        self.print_model_input_type()
        # 根据预训练模型获取输入序列的最大长度


    def __repr__(self):
        return "{}".format(self.__dict__.items())

    def get_model_max_length(self):
        # 获取模型名
        PLM = os.path.basename(self.pretrain_model_path)
        model_length_dict = {
            "mbart25" : 512,
            "mbart50" : 512,
            "chinese-macbert-large" : 512,
            "chinese-macbert-base" : 512,
            "BERT" : 512,
            "AnchiBERT" : 512,
            "GuwenBERT" : 512,
            "Longformer" : 1024,
            "mt5-base" : 512,
        }

        self.model_max_length = model_length_dict[PLM]
        self.length = self.model_max_length

        if "2_" in self.model_input_type:
            self.length = self.model_max_length * 2

        if "3_" in self.model_input_type:
            self.length = self.model_max_length * 3

        if "4" in self.model_input_type:
            self.length = self.model_max_length * 4

        if "5" in self.model_input_type:
            self.length = self.model_max_length * 5



    def print_model_input_type(self):
        if self.model_input_type == self.MODEL_INPUT_TYPE1:
            print("\t \t \t \t \t \t \t \t512（question + options + passage_1）")
        if self.model_input_type == self.MODEL_INPUT_TYPE2_1:
            print("\t \t \t \t \t \t \t \t512（question + options + passage_1）+ 512（passage）")
        if self.model_input_type == self.MODEL_INPUT_TYPE2_2:
            print("\t \t \t \t \t \t \t \t512（question + options）+ 512（passage_1）")
        if self.model_input_type == self.MODEL_INPUT_TYPE3_1:
            print("\t \t \t \t \t \t \t \t512（question + options）+ 512（passage_1）+ 512（passage_2）")
        if self.model_input_type == self.MODEL_INPUT_TYPE3_2:
            print("\t \t \t \t \t \t \t \t512（question + options + passage_1）+ ")
            print("\t \t\t \t \t \t \t \t512（question + options + passage_2）+ ")
            print("\t \t\t \t\t \t \t \t512（question + options + passage_3）+ ")
        if self.model_input_type == self.MODEL_INPUT_TYPE5:
            print("\t \t\t \t \t \t \t \t512（question + options + passage_1）+")
            print("\t \t\t \t\t \t \t \t512（question + options + passage_2）+")
            print("\t \t\t \t \t \t \t \t512（question + options + passage_3）+")
            print("\t \t\t \t \t \t \t \t512（question + options + passage_4）+")
            print("\t \t\t \t\t \t \t \t512（question + options + passage_5）")
