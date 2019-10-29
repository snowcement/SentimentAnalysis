#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/29 18:53
# @Author  : wutt
# @File    : mydataset.py

from torch.utils.data import Dataset
COUNT_LABEL0 = 761
COUNT_LABEL1 = 3590
COUNT_LABEL2 = 2914

def up_sampling():
    '''
    1    3590
    2    2914
    0     761
    :return:
    '''
    pass
class MyDataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, item):
        # EDA to augment data
        pass
    def __len__(self):
        pass