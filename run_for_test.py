#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 19:17
# @Author  : wutt
# @File    : run_for_test.py
# script for test
import torch
from model_util import load_model
import unittest
import os
import logging
import pandas as pd
import pickle

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.tokenization_xlnet import XLNetTokenizer
import args
from torch.utils.data import SequentialSampler, TensorDataset, DataLoader

from data_loader import select_field
from data_preprocess import SentiAnalysisProcessor, convert_examples_to_features
logger = logging.getLogger(__name__)


# TODO: 单样本分段特征
def convert_split_doc_to_feature(doc, max_seq_length, split_num, cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    '''
    将测试文本转化为测试特征
    :param doc: corpus for test
    :return: feature for predicting
    '''
    tokenizer = XLNetTokenizer.from_pretrained(os.path.join(args.ROOT_DIR, args.xlnet_model),
                                               do_lower_case=args.do_lower_case)
    tokens_a = tokenizer.tokenize(doc)

    skip_len = len(tokens_a) / split_num
    choices_features = []

    for i in range(split_num):
        context_tokens_choice = tokens_a[int(i * skip_len):int((i + 1) * skip_len)]

        tokens_b = None

        # Account for [CLS] and [SEP] with "- 2"
        if len(context_tokens_choice) > max_seq_length - 2:
            context_tokens_choice = context_tokens_choice[:(max_seq_length - 2)]

        tokens = context_tokens_choice + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        choices_features.append({'input_ids': input_ids,
                                 'input_mask': input_mask,
                                 'segment_ids': segment_ids})

        # logger.info("*** Example ***")
        # logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
        # logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
        # logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
        # logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))

    return choices_features

# TODO: 未分段特征
def convert_doc_to_feature(doc, max_seq_length, cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):



    '''
    使用Bert将测试文本转化为测试特征
    :param doc: corpus for test
    :return: feature for predicting
    '''
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.ROOT_DIR, args.VOCAB_FILE))
    tokens_a = tokenizer.tokenize(doc)
    tokens_b = None
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    logger.info("*** Example ***")
    logger.info("tokens: %s" % " ".join(
        [str(x) for x in tokens]))
    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    return input_ids, input_mask, segment_ids

def predict_test_xlnet(doc, model, device):
    features = convert_split_doc_to_feature(doc, args.max_seq_length, split_num=args.split_num,
                                            cls_token_at_end=bool(args.model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                            pad_on_left=bool(args.model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)

    all_input_ids = all_input_ids[None]
    all_segment_ids = all_segment_ids[None]
    all_input_mask = all_input_mask[None]

    with torch.no_grad():
        encode = model(all_input_ids.to(device), all_segment_ids.to(device), all_input_mask.to(device))
        encode = encode[0]
        predicts = model.predict(encode)

    print("预测结果：",args.labels[predicts[0]])
    return args.labels[predicts[0]]

def predict_test_bert(doc, model, device):
    '''
    根据测试文本生成分类结果
    :param doc:带预测语料
    :return:标注结果
    '''
    model.eval()
    features = convert_doc_to_feature(doc, args.max_seq_length,
                                      cls_token_at_end=bool(args.model_type in ['xlnet']),
                                      # xlnet has a cls token at the end
                                      cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                      pad_on_left=bool(args.model_type in ['xlnet']),  # pad on the left for xlnet
                                      pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)


    # 转化为model输入形式
    input_ids = torch.tensor([features.input_ids], dtype=torch.long)
    input_mask = torch.tensor([features.input_mask], dtype=torch.long)
    segment_ids = torch.tensor([features.segment_ids], dtype=torch.long)

    with torch.no_grad():
        bert_encode = model(input_ids.to(device), segment_ids.to(device), input_mask.to(device))
    bert_encode = bert_encode[0]  # 提取预测结果
    # #数值化的标签预测结果
    predicts = model.predict(bert_encode.detach().cpu().numpy())[0]
    print("预测结果：",args.labels[predicts[0]])
    return args.labels[predicts[0]]


class SentimentAnalysisTest(unittest.TestCase):
    def test_ner(self):
        model = load_model(os.path.join(args.ROOT_DIR, args.output_dir), args.model_type)
        doc = '西宁东方医院坑骗患者 医生夸大病情恐吓消费者太黑心!_访客创业网-正在为您跳转到访问页面.....\n如果您的浏览器没有自动跳转，请检查以下设置。(1)请确保浏览器没有禁止发送cookie。(2)请确保浏览器可以正常执行javascript脚本。(3)若使用ie浏览器，请使用ie9及以上版本。(4)确保本地时间的准确性。(5)请观察这个时间(2019-01-0220:56:15)若时间一直未变化，则是由于验证页面被缓存，可能是与cdn设置不兼容。'
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl')

        model.to(device)
        predict_test_bert(doc, model, device)
        #predict_test_xlnet(doc, model, device)

        # model = load_model(os.path.join(args.ROOT_DIR, args.output_dir), args.model_type)
        # if args.model_type == 'bert':
        #     f = predict_test_bert
        # elif args.model_type == 'xlnet':
        #     f = predict_test_xlnet
        # while True:
        #     doc = input("please input the doc for sentiment analysis: (type 'quit' if you want to quit)\n")
        #     if doc == 'quit':
        #         break
        #     else:
        #         #print("input >> " + doc)
        #         f(doc.encode('GBK').decode('GBK'), model)


def preprocess():
    '''
    针对测试集数据预测
    :return:
    '''
    model = load_model(os.path.join(args.ROOT_DIR, args.output_dir), args.model_type)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1

    model.to(device)

    processors = {"sentiment_analysis": SentiAnalysisProcessor}
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    examples = processor.get_test_examples(args.data_dir)

    tokenizer = XLNetTokenizer.from_pretrained(os.path.join(args.ROOT_DIR, args.xlnet_model),
                                               do_lower_case=args.do_lower_case)
    mode = 'test'
    try:
        if mode == 'test':
            with open(os.path.join(args.data_dir, args.TEST_FEATURE_FILE), 'rb') as f:  # TRAIN_FEATURE_FILE
                features = pickle.load(f)
    except:
        features = convert_examples_to_features(examples, args.max_seq_length, args.split_num, tokenizer,
                                                mode=mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                pad_on_left=bool(args.model_type in ['xlnet']),  # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)

    #数据集
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    sampler = SequentialSampler(dataset)#作用近似shuffle
    dataloader = DataLoader(dataset, sampler=sampler, batch_size = args.per_gpu_train_batch_size * max(1, n_gpu))
    model.eval()
    y_predicts = []
    for input_ids, input_mask, segment_ids in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, token_type_ids=segment_ids,
                           attention_mask=input_mask)[0]
            predicts = model.predict(logits)
        y_predicts.append(torch.from_numpy(predicts))
    eval_predicted = torch.cat(y_predicts, dim=0).cpu().numpy()

    df = pd.read_csv(os.path.join(args.data_dir, args.TEST_CORPUS_FILE))
    df['labels'] = eval_predicted
    df[['id','labels']].to_csv('./data/test_final.csv', sep=',', encoding='utf_8_sig', header=True, index=False)

if __name__ == "__main__":
    unittest.main()
    #preprocess()
