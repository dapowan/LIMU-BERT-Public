#!/bin/bash
nohup python -u pretrain.py v1 motion 20_120 -g 0 -s v1 > log/pretrain_base_motion.log &
nohup python -u pretrain.py v1 uci 20_120 -g 0 -s v1 > log/pretrain_base_uci.log &
nohup python -u pretrain.py v1 hhar 20_120 -g 0 -s v1 > log/pretrain_base_hhar.log &
nohup python -u pretrain.py v1 shoaib 20_120 -g 0 -s v1 > log/pretrain_base_shoaib.log &
