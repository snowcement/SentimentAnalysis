+ 2019-10-29-10-48-00
    + lr=5e-5
    + warmup_steps=2
    + gradient_accumulation_steps=1
    + 未进行DA(data augmentation)
    + weight_decay = 0.01
    + Bert model
+ 2019-10-29T16-04-44
    + lr=5e-5
    + warmup_steps=120------------------
    + gradient_accumulation_steps=1
    + 未进行DA(data augmentation)
    + weight_decay = 0.01
    + Bert model
+ 2019-10-29-14-35-00
    + lr=2e-5---------------------
    + warmup_steps=120
    + gradient_accumulation_steps=1
    + 未进行DA
    + weight_decay = 0.01
    + Bert model
+ 2019-10-29T15-01-36
    + lr=5e-5(加快training speed)
    + warmup_steps=120
    + gradient_accumulation_steps=4(变相增大batch size,直接增加batch size会超出显存报错)--------------
    + 未进行DA
    + weight_decay = 0.01
    + Bert model
+ 2019-10-29T18-27-17(better)
    + lr=2e-5-------------------
    + warmup_steps=120
    + gradient_accumulation_steps=4
    + 未进行DA
    + weight_decay = 0.01
    + Bert model
+ 2019-10-30T19-51-51
    + lr=2e-5
    + warmup_steps=120
    + gradient_accumulation_steps=4
    + 使用EDA对少量样本类进行DA------------------
        + 使用random swap
        + alpha_rs=0.15
    + weight_decay = 0.01
    + Bert model
+ 2019-10-30T22-43-43(没过拟合？)
    + lr=2e-5
    + warmup_steps=120
    + gradient_accumulation_steps=4
    + 使用EDA对少量样本类进行DA
    + weight_decay = 0.1----------------------
    + Bert model
+ 2019-11-10T09-46-40
    + adam_epsilon=1e-8
    + train_batch = 1
    + eval_batch = 4-------------
    + lr=2e-5
    + warmup_steps=120
    + gradient_accumulation_steps=2
    + 使用EDA对少量样本类进行DA
    + XLnet model-------------
+ 2019-11-10T09-46-40
    + adam_epsilon=1e-6-----------
    + train_batch = 1
    + eval_batch = 64-------------
    + lr=5e-5-----------
    + warmup_steps=120
    + gradient_accumulation_steps=2
    + 使用EDA对少量样本类进行DA
    + XLnet model-------------
+ 2019-11-10T22-40-35
    + adam_epsilon=1e-6-----------
    + train_batch = 1
    + eval_batch = 64-------------
    + lr=2e-5-----------
    + warmup_steps=120
    + gradient_accumulation_steps=2
    + 使用EDA对少量样本类进行DA
    + XLnet model
    + split_num = 10
    + max_seq_length = 150(之前均为200)
+ 2019-11-10T22-40-35
    + adam_epsilon=1e-6-----------
    + train_batch = 1
    + eval_batch = 64-------------
    + lr=5e-6-----------
    + warmup_steps=120
    + gradient_accumulation_steps=2
    + 使用EDA对少量样本类进行DA
    + XLnet model
    + split_num = 10
    + max_seq_length = 150
