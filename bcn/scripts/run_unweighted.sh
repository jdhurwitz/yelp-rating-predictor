#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py --cuda --gpu 0 --task yelp --save_path '../bcn_output_unweighted/' --max_words 80000 --emtraining