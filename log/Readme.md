+ 2019-10-29-10-48-00
    + lr=5e-5
    + warmup_steps=2
    + gradient_accumulation_steps=1
+ 2019-10-29-14-35-00
    + lr=2e-5
    + warmup_steps=120
    + gradient_accumulation_steps=1
+ 2019-10-29T15-01-36
    + lr=5e-5(加快training speed)
    + warmup_steps=120
    + gradient_accumulation_steps=4(变相增大batch size,直接增加batch size会超出显存报错)