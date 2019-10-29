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
from pytorch_transformers import AdamW, WarmupLinearSchedule
from datetime import datetime

from model_util import save_model
from Bert_SenAnalysis import Bert_SenAnalysis
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


def fit(model, training_iter, eval_iter, num_epoch, pbar, num_train_steps, device, n_gpu, verbose=1):
    # ------------------结果可视化------------------------
    if args.local_rank in [-1, 0]:
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        tb_writer = SummaryWriter('log/%s'%TIMESTAMP)
    # ---------------------优化器-------------------------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    t_total = num_train_steps

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
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

    # train_losses = []
    # eval_losses = []
    # train_accuracy = []
    # eval_accuracy = []

    # history = {
    #     "train_loss": train_losses,
    #     "train_acc": train_accuracy,
    #     "eval_loss": eval_losses,
    #     "eval_acc": eval_accuracy
    # }
    tr_loss, logging_loss = 0.0, 0.0
    # ------------------------训练------------------------------
    best_f1 = 0
    start = time.time()
    global_step = 0
    set_seed(args, n_gpu)  # Added here for reproductibility (even between python 2 and 3)
    for e in range(num_epoch):
        model.train()
        for step, batch in enumerate(training_iter):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            bert_encode = model(**inputs)
            bert_encode = bert_encode[0]#提取预测结果
            train_loss = model.loss_fn(bert_encode=bert_encode,
                                       labels=inputs['labels'])

            if n_gpu > 1:
                train_loss = train_loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                train_loss = train_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += train_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    #     results = evaluate(args, model, tokenizer)
                    #     for key, value in results.items():
                    #         tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            predicts = model.predict(bert_encode.detach().cpu().numpy())
            label_ids = inputs['labels'].view(1, -1)
            label_ids = label_ids[label_ids != -1]
            label_ids = label_ids.cpu().numpy()

            train_acc, train_rec, f1 = model.acc_rec_f1(predicts, label_ids)
            pbar.show_process(train_acc, train_rec, train_loss.item(), f1, time.time() - start, step)
    # -----------------------验证----------------------------
        model.eval()
        count = 0
        y_predicts, y_labels = [], []
        eval_loss, eval_acc, eval_f1 = 0, 0, 0

        with torch.no_grad():
            for step, batch in enumerate(eval_iter):
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3]}
                bert_encode = model(**inputs)
                bert_encode = bert_encode[0]  # 提取预测结果
                eval_los = model.loss_fn(bert_encode=bert_encode, labels=inputs['labels'])
                eval_loss = eval_los + eval_loss
                count += 1
                predicts = model.predict(bert_encode.detach().cpu().numpy())
                y_predicts.append(torch.from_numpy(predicts))

                label_ids = inputs['labels'].view(1, -1)
                label_ids = label_ids[label_ids != -1]
                y_labels.append(label_ids)

            eval_predicted = torch.cat(y_predicts, dim=0).cpu().numpy()
            eval_labeled = torch.cat(y_labels, dim=0).cpu().numpy()

            eval_acc, eval_rec, eval_f1 = model.acc_rec_f1(eval_predicted, eval_labeled)
            model.class_report(eval_predicted, eval_labeled)

            logger.info(
                '\n\nEpoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - eval_f1:%4f\n'
                % (e + 1,
                   train_loss.item(),
                   eval_loss.item() / count,
                   train_acc,
                   eval_acc,
                   eval_f1))

            # 保存最好的模型
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                save_model(model, args.output_dir)

            if args.local_rank in [-1, 0]:
                tb_writer.add_scalar('train_loss_per_epoch', train_loss.item(), e)
                tb_writer.add_scalar('train_acc', train_acc, e)
                tb_writer.add_scalar('eval_loss', eval_loss.item() / count, e)
                tb_writer.add_scalar('eval_acc', eval_acc, e)

            # if e % verbose == 0:
            #     train_losses.append(train_loss.item())
            #     train_accuracy.append(train_acc)
            #     eval_losses.append(eval_loss.item() / count)
            #     eval_accuracy.append(eval_acc)


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

    train_iter, num_train_steps = load_and_cache_examples(mode='train',
                                                          train_batch_size=train_batch_size,
                                                          eval_batch_size=eval_batch_size)
    eval_iter = load_and_cache_examples(mode='dev',
                                        train_batch_size=train_batch_size,
                                        eval_batch_size=eval_batch_size)

    epoch_size = num_train_steps * train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs

    pbar = ProgressBar(epoch_size=epoch_size,
                       batch_size=train_batch_size)
    model = Bert_SenAnalysis.from_pretrained(args.bert_model, num_tag = len(args.labels))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    fit(model = model,
        training_iter=train_iter,
        eval_iter=eval_iter,
        num_epoch=args.num_train_epochs,
        pbar=pbar,
        num_train_steps=num_train_steps,
        device=device,
        n_gpu=n_gpu,
        verbose=1)

if __name__ == "__main__":
    main()


#--train_file=/home/webapp/newspace/script/wtt/pytorch-transformers/examples/tests_samples/SQUAD/dev-v2.0-small.json
# --predict_file=/home/webapp/newspace/script/wtt/pytorch-transformers/examples/tests_samples/SQUAD/dev-v2.0-small.json
# # --model_name=bert-base-uncased --output_dir=/home/webapp/newspace/script/wtt/pytorch-transformers/examples/tests_samples/temp_dir
# --max_steps=10 --warmup_steps=2 --do_train --do_eval --version_2_with_negative --learning_rate=1e-4
# --per_gpu_train_batch_size=2 --per_gpu_eval_batch_size=1 --overwrite_output_dir --seed=42 --model_type=bert
# --model_name_or_path=/home/webapp/newspace/script/wtt/pytorch-transformers/model_and_config/bert_base_uncased/bert-base-uncased-pytorch_model.bin
# --tokenizer_name=/home/webapp/newspace/script/wtt/pytorch-transformers/model_and_config/bert_base_uncased/bert-base-uncased-vocab.txt
# --config_name=/home/webapp/newspace/script/wtt/pytorch-transformers/model_and_config/bert_base_uncased/bert-base-uncased-config.json