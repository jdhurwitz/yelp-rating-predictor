#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python main.py --cuda --gpu 0 --task yelp --data '../../data/yelp/gm_data/' --save_path '../../model8_output/' --max_words 80000 --emtraining --pos