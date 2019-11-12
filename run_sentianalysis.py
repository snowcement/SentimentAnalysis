#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/16 15:17
# @Author  : wutt
# @File    : run_sentianalysis.py
import torch
import os
import logging
import random
import numpy as np
import time
from tensorboardX import SummaryWriter
from pytorch_transformers import AdamW, WarmupLinearSchedule, XLNetConfig
from datetime import datetime
from itertools import cycle
from tqdm import tqdm
from model_util import save_model
from Bert_SenAnalysis import Bert_SenAnalysis
from Xlnet_SenAnalysis import XLNet_SenAnalysis
from data_loader import load_and_cache_examples
from progress_util import ProgressBar
import args#配置文件中的变量

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',level=logging.INFO)

def set_seed(args, n_gpu):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def fit(model, training_iter, eval_iter, num_train_steps, device, n_gpu, verbose=1):
    # ------------------结果可视化------------------------
    if args.local_rank in [-1, 0]:
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        tb_writer = SummaryWriter('log/%s'%TIMESTAMP)
    # ---------------------优化器-------------------------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    t_total = num_train_steps

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)#int(t_total*args.warmup_proportion)
    # ---------------------GPU半精度fp16-----------------------------
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # ---------------------模型初始化----------------------
    model.to(device)
    tr_loss, logging_loss = 0.0, 0.0
    # ------------------------训练------------------------------
    best_f1 = 0
    #start = time.time()
    global_step = 0
    set_seed(args, n_gpu)  # Added here for reproductibility (even between python 2 and 3)
    bar = tqdm(range(t_total), total = t_total)
    nb_tr_examples, nb_tr_steps = 0, 0

    for step in bar:
        model.train()
        batch = next(training_iter)
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                  # XLM don't use segment_ids
                  'labels': batch[3]}
        encode = model(**inputs)
        encode = encode[0]#提取预测结果
        loss = model.loss_fn(encode, labels=inputs['labels'])

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            #torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        tr_loss += loss.item()
        train_loss = round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
        bar.set_description("loss {}".format(train_loss))
        nb_tr_examples += inputs['input_ids'].size(0)
        nb_tr_steps += 1

        if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()
            global_step += 1

        if (step + 1) %(args.eval_steps*args.gradient_accumulation_steps)==0:
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            logger.info("***** Report result *****")
            logger.info("  %s = %s", 'global_step', str(global_step))
            logger.info("  %s = %s", 'train loss', str(train_loss))


        if args.local_rank in [-1, 0] and \
                args.do_eval and (step+1)%(args.eval_steps*args.gradient_accumulation_steps)==0:

            # -----------------------验证----------------------------
            model.eval()
            y_predicts, y_labels = [], []
            eval_loss, eval_acc, eval_f1 = 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for _, batch in enumerate(eval_iter):
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}
                with torch.no_grad():
                    encode = model(**inputs)
                    encode = encode[0]  # 提取预测结果
                    eval_los = model.loss_fn(encode, labels=inputs['labels'])

                    predicts = model.predict(encode)#.detach().cpu().numpy()

                nb_eval_examples += inputs['input_ids'].size(0)
                nb_eval_steps += 1
                eval_loss += eval_los.mean().item()
                y_predicts.append(torch.from_numpy(predicts))

                labels = inputs['labels'].view(1, -1)
                labels = labels[labels != -1]
                y_labels.append(labels)

            eval_loss = eval_loss / nb_eval_steps
            eval_predicted = torch.cat(y_predicts, dim=0).cpu().numpy()
            eval_labeled = torch.cat(y_labels, dim=0).cpu().numpy()

            eval_f1 = model.acc_rec_f1(eval_predicted, eval_labeled)#eval_acc, eval_rec,

            logger.info(
                '\n\nglobal_step %d - train_loss: %4f - eval_loss: %4f - eval_f1:%4f\n'
                % (global_step,
                   train_loss,
                   eval_loss,
                   eval_f1))

            # 保存最好的模型
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                save_model(model, args.output_dir)

            if args.local_rank in [-1, 0]:
                tb_writer.add_scalar('train_loss', train_loss, step)#.item()
                tb_writer.add_scalar('eval_loss', eval_loss, step)#.item() / count
                tb_writer.add_scalar('eval_f1', eval_f1, step)#eval_acc

            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

def main():
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # ------------------判断CUDA模式----------------------
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1

    #produce data
    train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpu)
    eval_batch_size = args.per_gpu_eval_batch_size * max(1, n_gpu)

    train_iter = load_and_cache_examples(mode='train',train_batch_size=train_batch_size,
                                                          eval_batch_size=eval_batch_size)
    eval_iter = load_and_cache_examples(mode='dev',
                                        train_batch_size=train_batch_size,
                                        eval_batch_size=eval_batch_size)

    #epoch_size = num_train_steps * train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs

    # pbar = ProgressBar(epoch_size=epoch_size,
    #                    batch_size=train_batch_size)

    if args.model_type == 'bert':
        model = Bert_SenAnalysis.from_pretrained(args.bert_model, num_tag = len(args.labels))
    elif args.model_type == 'xlnet':
        config = XLNetConfig.from_pretrained(args.xlnet_model, num_labels = len(args.labels))
        model = XLNet_SenAnalysis.from_pretrained(args.xlnet_model, config=config)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    train_iter = cycle(train_iter)
    fit(model = model,
        training_iter=train_iter,
        eval_iter=eval_iter,
        #train_steps=args.train_steps,
        #pbar=pbar,
        num_train_steps=args.train_steps,#num_train_steps,
        device=device,
        n_gpu=n_gpu,
        verbose=1)

if __name__ == "__main__":
    main()