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

from pytorch_transformers.tokenization_bert import BertTokenizer
import args
logger = logging.getLogger(__name__)
#logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',level=logging.INFO)

def convert_doc_to_feature(doc, max_seq_length, cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    '''
    将测试文本转化为测试特征
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


def predict_test(doc, model):
    '''
    根据测试文本生成标注结果
    :param doc:带预测语料
    :return:标注结果
    '''
    input_ids, input_mask, segment_ids = convert_doc_to_feature(doc, args.max_seq_length,
                                            cls_token_at_end=bool(args.model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                            pad_on_left=bool(args.model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    # 转化为model输入形式
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_mask = torch.tensor([input_mask], dtype=torch.long)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long)


    bert_encode = model(input_ids, segment_ids, input_mask)
    bert_encode = bert_encode[0]  # 提取预测结果
    #数值化的标签预测结果
    predicts = model.predict(bert_encode.detach().cpu().numpy())[0]
    print("预测结果：",args.labels[predicts])
    return args.labels[predicts]


#class SentimentAnalysisTest(unittest.TestCase):
    # def test_ner(self):
    #     #doc(label:1) = '【收藏】全新推出:f1签名插画海报。绝对不容错过!-你一定还记得去年我们首次发售的保时捷wec签名插画海报。你知道吗?这款收藏品出乎我们预料之外，不到2个月就基本售罄了。后来，为了纪念我们傻豹豹赛车工作室微博开通七周年，我们在2018年1月又再次推出了限量版的保时捷919模型底座，带证书的10套底座也是在一周内一下子销售一空......或许你两者都已经收藏了，或许你还在持观摩的态度，等待傻豹豹推出让自己心动的新产品.....【收藏】赛车迷们!你的专属圣诞节礼物在这里现在，你无须再等待了.....从在2018f1赛季开始，傻豹豹赛车工作室再次与damingworkroom合作推出全新的f1赛车签名插画海报。第一批首先推出的是梅赛德斯车手汉密尔顿，迈凯伦车手阿隆索，法拉利车手维特尔，以及在今年上海站获胜的红牛车队的两位车手里卡多和维斯塔潘。每一张作品都由daming亲自操刀绘制制作，结合傻豹豹在f1上海站赛车现场获得的车手的亲笔签名印刷，采用240g的特种高感纸印刷，600dpi分辨率完美还原色彩和画面细节。是一款不可多得的f1周边与收藏品。从这次开始，我们给作品添加了产品序号——00x。去年我们推出的两分保时捷wec签名插画分别是:#001保时捷919，以及#002保时捷911gte，所以从新的这批f1作品里，你会发现按顺序依次是#003，#004......要想知道自己是否收齐了我们的作品，记得核对这个产品编码哦!与以往一样，每一张作品都是由傻豹豹赛车工作室亲自获得签名的。我们一起看看此次的签名里，有没有你值得崇拜的车手。当你看到这篇文章内容的时候，以上的作品都已经悄然上架了。你可以通过下面的链接进行购买。这一次我们为了大家收藏方便使用了两种尺寸的插画，分别是a3以及a4。a3更适合挂墙，而a4更适合放你的桌面。(上面为a3，下面为a4)目前我们的定价是a3是128元/张包顺丰，a4是98元/每张包顺丰。但是如果你计划购入一套4张，我们会给你一个88折优惠，分别是a4340元包邮，以及a3450元包邮的价格。tb的链接如下:(请复制url到你的浏览器上面直接打开，或者复制后在tbapp打开)'
    #     #predict_test(doc)#
    #     # model = load_model(os.path.join(args.ROOT_DIR, args.output_dir))
    #     # doc = '有一种生活，叫住在临沂老城区!- 临沂是一座很容易让人爱上的城市，尤其是老城区。对于一个每城市来说，新城总是千篇一律，只有老城区会保留着最本真的生活态度。生活在临沂老城，一街一巷慢慢寻找历史的影子，看着邻里话家常，听着声声叫卖，浓浓的市井风情分外动人。七遇提起老城区，首先浮现在脑海的是略显狭窄，笔直向前方延伸的老路和路边蔽天的梧桐，仿佛能看到时间静止的纹路。可在我看来，新城区长着同一副面孔，每个城市大同小异，而老城区的魅力正是在这一条条老街小巷中。对于临沂老城区的街坊们来说，幽静的巷子是儿时的回忆，几十年如一日的时光，经由他们留下道道刻痕。老城区里的平房已经越来越少见，多的可能是老式居民楼或二层小楼，不复当年繁华，带着浓重的岁月痕迹，很多人就生在那里、长在那里，现在依旧住在那里。一条小巷隔出两重世界:巷口那条街是嘈杂喧闹的，而小巷深处的家是静谧安宁的。一切都变得那么舒缓，那么心定气闲。沂州路、沂蒙路、洗砚池街、考棚街、关帝庙街、关东街每一个名字背后都有着长长的故事。但随着城市的发展记忆中的景象也慢慢的消失，记忆中的生活生活境况越来越少见，可能正是这样，才会有人对老城充满记念牵挂。新城是临沂人的希望，而老城是临沂人灵魂的归属。很多人不喜欢“市井”这个词，觉得会拉低逼格，但生活在老城的临沂人不会。它代表了一种生活方式。那些市井味浓的地方，反而是老临沂们爱待、爱去的地方。临沂的商场越来越多，道路越来越宽，吃喝玩乐的选择越来越多，对比起来，不太容易被提起的那些菜场、水产市场、副食一条街等地方才有市井味儿，也更吸引他们。它没有响当当的名声，一点也不显山露水，普通得不能再普通，平民得不能再平民，纯粹地就是老百姓过日子的地方。也总有很多临沂人来光顾，为了选到心仪的那一物，他们愿意早起、愿意走很远的路这种爱好，是不会因为钱多钱少而改变的。他们喜欢在空气湿漉漉的清晨，穿过幽长的小巷，穿过香味阵阵的小食店，去菜市场挑拣新鲜的蔬菜，带回一把青翠欲滴的青菜，一筐红光满面的番茄他们喜欢去老城某个巷子，一碗白粥、一碗糁汤、几根油条、几笼汤包，那些几十年的店，早晚挤满了熟悉的面孔，天天这样吃早餐吃了大半辈子临沂老城的市井气是一种刻在骨子里的生活味道。临沂有越来越多的餐厅、美食，装修逼格之高，菜品之精致，可是老临沂人不会每天光顾，想品尝老临沂的味道，还是得去这些老巷子里寻找。新城区人羡慕着老城人的“好口福”。炒鸡、糁汤、羊汤这些最最正宗的味道一定是在古城里。数不尽的美食、小吃，几十年的老字号比比皆是。很多小时候就有的店，临沂人从小吃到大，时光在变，不变的是口感，是情怀。想把临沂吃透呀，没吃遍老城可不行!经营这这些老味道的店家在老城住了几十年，习惯了老城的节奏，也习惯了来来往往的老面孔，他们眷恋着老城，所以想在新城区品尝到他们的味道，那可真的是太难了!在老城生活久了，总能从小巷角落里找到门店不起眼，但味道绝对赞的美食。 每天早晨，上班路上，在那些个巷子里，你总能看到，围着一圈的人，他们是公司的白领，附近的居民，他们都在一起，排队。老城里的临沂人都比较怀旧，特别是小时候的情景，都在脑袋里。那时候，北城还是很遥远，记忆很窄，窄得只有这一片老城区....那时候，家里还没有空调，过夏天么，坐在院子里乘风凉，自来水往院子里一泼，竹板凳子一坐，吃着井水冰镇过的西瓜，特别凉爽。那时候，最开心的就是爸妈带着去逛大集!街上熙熙攘攘的都是人，却只觉得热闹，一路逛逛缠着爸妈买各种小吃，开心的不得了。那时候，住在巷子里，大家几十年的老邻居，邻里之间关系都很熟，很多都是“发小”。记忆里的巷子很长很深，在巷子里穿来窜去打闹的日子仿佛还在昨天!那时候，巷子里还有叫卖声。一辆自行车，一个木箱，仿佛世界都装在那个木箱里。许久没有再见过这样卖棒冰了，以后我们的孩子也不会再见到了吧。有一种生活，叫住在临沂老城。时间再怎么变，只要走在老街巷子里，就感觉好像时光不曾远去。'
    #     # predict_test(doc.encode('utf-8').decode('utf-8'), model)#(label:0)
    #     model = load_model(os.path.join(args.ROOT_DIR, args.output_dir))
    #     while True:
    #         doc = input("please input the doc for sentiment analysis: (type 'quit' if you want to quit)\n")
    #         if doc == 'quit':
    #             break
    #         else:
    #             #print("input >> " + doc)
    #             predict_test(doc.encode('GBK').decode('GBK'), model)


def preprocess():
    model = load_model(os.path.join(args.ROOT_DIR, args.output_dir))
    datadf = pd.read_csv('./data/Test_DataSet.csv')
    # title,content均包含nan,将title,content二者内容拼接,剔除空行
    datadf['content'] = datadf['title'].str.cat(datadf['content'], sep='-')
    datadf.drop(['title'], axis=1, inplace=True)
    datadf.info()
    print(datadf.isnull().any())
    labels = []
    for doc in list(datadf['content']):
        labels.append(predict_test(doc, model))
    df = pd.DataFrame({'测试数据id': datadf['id'], '情感极性': labels})
    df.to_csv('./data/default7648034-final.csv', sep=',', encoding='utf_8_sig', header=True, index=False)

if __name__ == "__main__":
    #unittest.main()
    preprocess()
