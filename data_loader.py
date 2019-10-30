#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 14:55
# @Author  : wutt
# @File    : data_loader.py
# examples into features, dataset, dataloaaer
import torch
from data_preprocess import SentiAnalysisProcessor, convert_examples_to_features
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from pytorch_transformers import BertConfig, BertTokenizer
import os
import logging
import pickle

import args

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',level=logging.INFO)

def init_params():
    processors = {"sentiment_analysis": SentiAnalysisProcessor}
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    tokenizer = BertTokenizer(vocab_file=args.VOCAB_FILE)
    return processor, tokenizer


def load_and_cache_examples(mode, train_batch_size, eval_batch_size):
    '''
    :param mode: "train","dev"
    :return:
    '''
    """构造迭代器"""
    processor, tokenizer = init_params()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
        #t_total
        num_train_steps = int(
            len(examples) / train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        batch_size = train_batch_size
    elif mode == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
        batch_size = eval_batch_size
    else:
        raise ValueError("Invalid mode %s" % mode)

    label_list = processor.get_labels()
    output_mode = 'classification'
    #特征
    try:
        if mode == "train":
            with open(os.path.join(args.data_dir, args.TRAIN_US_FEATURE_FILE), 'rb') as f:#TRAIN_FEATURE_FILE
                features = pickle.load(f)
        else:
            with open(os.path.join(args.data_dir, args.DEV_US_FEATURE_FILE), 'rb') as f:#DEV_FEATURE_FILE
                features = pickle.load(f)
    except:
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode, mode=mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    #数据集
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if mode == "train":
        sampler = RandomSampler(dataset)#作用近似shuffle
    elif mode == 'dev':
        sampler = SequentialSampler(dataset)
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    if mode == "train":
        return iterator, num_train_steps
    elif mode == 'dev':
        return iterator
    else:
        raise ValueError("Invalid mode %s" % mode)





