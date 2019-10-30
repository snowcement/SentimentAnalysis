#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 14:32
# @Author  : wutt
# @File    : args.py
# 超参数设置，变量定义
ROOT_DIR = '/home/haiyun/script/wtt/sentimentanalysis/'
TRAIN_CORPUS_FILE = "train.csv"#原始训练集数据
DEV_CORPUS_FILE = "dev_matched.csv"#原始验证集数据
TRAIN_US_CORPUS_FILE = "train_us.csv"#上采样后的训练集数据
DEV_US_CORPUS_FILE = "dev_matched_us.csv"#上采样后的验证集数据
TRAIN_FEATURE_FILE = "train_features.pkl"
DEV_FEATURE_FILE = "dev_features.pkl"
TRAIN_US_FEATURE_FILE = "train_features_us.pkl"
DEV_US_FEATURE_FILE = "dev_features_us.pkl"
VOCAB_FILE = "bert_wwm_chinese/vocab.txt"#Pretrained tokenizer name or path if not the same as model_name
data_dir = './data/'#The input data dir. Should contain the .csv files (or other data files) for the task.
model_type = 'bert'#Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())
labels = ['0','1','2']#['正面','中性','负面']
bert_model = 'bert_wwm_chinese/'#Path to pre-trained model or shortcut name selected in the list
task_name = 'sentiment_analysis'#The name of the task to train
output_dir = "data/output"#The output directory where the model predictions and checkpoints will be written.
#config_name = "%s/bert_wwm_chinese/bert_config.json"%path#Pretrained config name or path if not the same as model_name
cache_dir = ''#Where do you want to store the pre-trained models downloaded from s3
max_seq_length = 512#The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
do_train = True#Whether to run training.
do_eval = True#Whether to run eval on the dev set.
evaluate_during_training = False#Rul evaluation during training at each logging step.
do_lower_case = True#Set this flag if you are using an uncased model.
per_gpu_train_batch_size = 8#Batch size per GPU/CPU for training.
per_gpu_eval_batch_size = 8#Batch size per GPU/CPU for evaluation.
gradient_accumulation_steps = 4#Number of updates steps to accumulate before performing a backward/update pass.
learning_rate = 2e-5#5e-5...The initial learning rate for Adam.
alpha_rs = 0.15#random swap in eda
weight_decay = 0.0#Weight decay if we apply some.
adam_epsilon = 1e-8#Epsilon for Adam optimizer.
max_grad_norm = 1.0#Max gradient norm.
num_train_epochs = 5#Total number of training epochs to perform.
#max_steps = -1#If > 0: set total number of training steps to perform. Override num_train_epochs.
warmup_steps = 120#Linear warmup over warmup_steps.
logging_steps = 50#Log every X updates steps.
#save_steps = 50#Save checkpoint every X updates steps.
eval_all_checkpoints = False#Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number
no_cuda = False#Avoid using CUDA when available
overwrite_output_dir = False#Overwrite the content of the output directory
overwrite_cache = False#Overwrite the cached training and evaluation sets
seed = 42#random seed for initialization
fp16 = False#Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit
fp16_opt_level = 'O1'#For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html
local_rank = -1#For distributed training: local_rank
server_ip = ''#For distant debugging.
server_port = ''#For distant debugging.
