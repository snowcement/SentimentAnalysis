#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 14:02
# @Author  : wutt
# @File    : Bert_SenAnalysis.py
# 情感分类定义
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers import BertPreTrainedModel,BertModel
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report

class Bert_SenAnalysis(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self,
                 config,
                 num_tag):
        super(Bert_SenAnalysis, self).__init__(config)
        self.num_labels = num_tag

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_tag)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        return outputs

    def loss_fn(self, bert_encode, labels):
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(bert_encode.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(bert_encode.view(-1, self.num_labels), labels.view(-1))

        return loss

    def acc_rec_f1(self, preds, labels):
        # correct = np.sum((labels==preds).astype(int))
        # acc = correct/preds.shape[0]
        # acc = accuracy_score(y_true=labels, y_pred=preds)  # simple_accuracy(preds, labels)
        # rec = recall_score(y_true=labels, y_pred=preds, average="macro")
        f1 = f1_score(y_true=labels, y_pred=preds, labels=[0,1,2],average="macro")
        return f1   # "acc_and_f1": (acc + f1) / 2,


    def predict(self, bert_encode):
        '''
        classification
        :param bert_encode:
        :return:
        '''
        bert_encode = nn.functional.softmax(bert_encode, -1)
        bert_encode = bert_encode.detach().cpu().numpy()
        if np.argmax(bert_encode, axis=1).any() not in [0,1,2]:
            print('attention:',np.argmax(bert_encode, axis=1))#按行取最大值，返回最大值所在的索引
        return np.argmax(bert_encode, axis=1)
