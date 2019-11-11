#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/9 19:33
# @Author  : wutt
# @File    : Xlnet_SenAnalysis.py.py
# 利用xlnet进行情感分类定义
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers import XLNetPreTrainedModel, XLNetModel
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report
import numpy as np
import args

try:
    from torch.nn import Identity
except ImportError:
    # Older PyTorch compatibility
    class Identity(nn.Module):
        r"""A placeholder identity operator that is argument-insensitive.
        """
        def __init__(self, *args, **kwargs):
            super(Identity, self).__init__()

        def forward(self, input):
            return input


class SequenceSummary(nn.Module):
    r""" Compute a single vector summary of a sequence hidden states according to various possibilities:
        Args of the config class:
            summary_type:
                - 'last' => [default] take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj: Add a projection after the vector extraction
            summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_activation: 'tanh' => add a tanh activation to the output, Other => no activation. Default
            summary_first_dropout: Add a dropout before the projection and activation
            summary_last_dropout: Add a dropout after the projection and activation
    """
    def __init__(self, config):
        super(SequenceSummary, self).__init__()

        self.summary_type = config.summary_type if hasattr(config, 'summary_use_proj') else 'last'
        if self.summary_type == 'attn':
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = Identity()
        if hasattr(config, 'summary_activation') and config.summary_activation == 'tanh':
            self.activation = nn.Tanh()

        self.first_dropout = Identity()
        if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hidden_states, cls_index=None):
        """ hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2]-1, dtype=torch.long)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim()-1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2) # shape (bsz, XX, hidden_size)
        elif self.summary_type == 'attn':
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


class XLNet_SenAnalysis(XLNetPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **mems**:
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` input above). Can be used to speed up sequential decoding and attend to longer context.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        model = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(XLNet_SenAnalysis, self).__init__(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(args.lstm_dropout)
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(args.lstm_hidden_size*2, config.num_labels)
        self.W = []
        self.gru = []
        for i in range(args.lstm_layers):
            self.W.append(nn.Linear(args.lstm_hidden_size*2, args.lstm_hidden_size*2))
            self.gru.append(nn.GRU(config.hidden_size if i ==0 else args.lstm_hidden_size*4,\
                                   args.lstm_hidden_size, num_layers=1, bidirectional = True,\
                                   batch_first = True).cuda())
        self.W = nn.ModuleList(self.W)
        self.gru = nn.ModuleList(self.gru)
        self.init_weights()
        #self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None,
                labels=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_input_mask = input_mask.view(-1, token_type_ids.size(-1)) if input_mask is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_mems = mems.view(-1, mems.size(-1)) if mems is not None else None
        flat_perm_mask = perm_mask.view(-1, perm_mask.size(-1)) if perm_mask is not None else None
        flat_target_mapping = target_mapping.view(-1, target_mapping.size(-1)) if target_mapping is not None else None
        flat_head_mask = head_mask.view(-1, head_mask.size(-1)) if head_mask is not None else None

        transformer_outputs = self.transformer(flat_input_ids, token_type_ids=flat_token_type_ids,
                                               input_mask=flat_input_mask, attention_mask=flat_attention_mask,
                                               mems=flat_mems, perm_mask=flat_perm_mask, target_mapping=flat_target_mapping,
                                               head_mask=flat_head_mask)
        output = transformer_outputs[0][:,-1,:]
        output = output.reshape(input_ids.size(0), input_ids.size(1), -1)

        for w, gru in zip(self.W, self.gru):
            try:
                gru.flatten_parameters()
            except:
                pass
            output, hidden = gru(output)
            output = self.dropout(output)

        hidden = hidden.permute(1, 0, 2).reshape(input_ids.size(0), -1).contiguous()
        logits = self.logits_proj(hidden)

        # output = self.sequence_summary(output)
        # logits = self.logits_proj(output)

        outputs = (logits,)#(logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it
        return outputs

    def loss_fn(self, logits, labels):
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = loss#(loss,) + outputs

        return outputs  # return (loss), logits, mems, (hidden states), (attentions)

    def acc_rec_f1(self, preds, labels):
        # correct = np.sum((labels==preds).astype(int))
        # acc = correct/preds.shape[0]
        # acc = accuracy_score(y_true=labels, y_pred=preds)  # simple_accuracy(preds, labels)
        # rec = recall_score(y_true=labels, y_pred=preds, average="macro")
        f1 = f1_score(y_true=labels, y_pred=preds, labels=[0,1,2],average="macro")
        return f1   # "acc_and_f1": (acc + f1) / 2,


    def predict(self, xlnet_output):
        '''
        classification
        :param xlnet_output:
        :return:
        '''
        xlnet_output = nn.functional.softmax(xlnet_output, -1)
        xlnet_output = xlnet_output.detach().cpu().numpy()
        if np.argmax(xlnet_output, axis=1).any() not in [0,1,2]:
            print('attention:',np.argmax(xlnet_output, axis=1))
        return np.argmax(xlnet_output, axis=1)#按行取最大值，返回最大值所在的索引