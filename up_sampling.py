#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/29 18:53
# @Author  : wutt
# @File    : mydataset.py
# https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py
# without Synonym replacement and Random insertion

import numpy as np
import random

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):
    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words
    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_rs=0.1):#, p_rd=0.1, num_aug=9):
    '''
    数据增强，这里只使用random swap策略
    :param sentence:
    :param alpha_rs:swap次数占总句长的比例
    :return:
    '''
    words = list(sentence)
    num_words = len(words)

    n_rs = max(1, int(alpha_rs * num_words))

    # random swap
    a_words = random_swap(words, n_rs)
    sentence = ''.join(a_words)

    # random deletion
    # a_words = random_deletion(words, p_rd)
    # sentence = ''.join(a_words)

    return sentence


def split_corpus(indexes, labels):
    '''
    根据labels不同划分corpus
    :param indexes:
    :param labels:
    :return:
    '''
    indexes_0, indexes_1, indexes_2 = [],[],[]
    for i, label in enumerate(labels):
        if label == 0:
            indexes_0.append(indexes[i])
        elif label == 1:
            indexes_1.append(indexes[i])
        else:
            indexes_2.append(indexes[i])
    return indexes_0, indexes_1, indexes_2

def up_sampling(indexes, tokens, labels):
    '''
    1    3590
    2    2914
    0     761
    :param: index(list):original index
    :param: tokens(list):original news corpus
    :param: labels(list):original sentimental labelss
    :return:
    '''
    indexes_0, indexes_1, indexes_2 = split_corpus(indexes, labels)
    print('Sample size with label 0: ', len(indexes_0))
    print('Sample size with label 1: ', len(indexes_1))
    print('Sample size with label 2: ', len(indexes_2))
    # 样本数量少的类别样本补齐
    indexes_tl = indexes_1 + np.random.choice(indexes_0, len(indexes_1)).tolist() + \
             np.random.choice(indexes_2, len(indexes_1)).tolist()
    random.seed(888)
    random.shuffle(indexes_tl)

    tokens = np.array(tokens)[indexes_tl].tolist()
    labels = np.array(labels)[indexes_tl].tolist()

    return tokens, labels