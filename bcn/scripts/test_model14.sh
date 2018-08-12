#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python test.py --cuda --gpu 0 --task yelp --data '../data/yelp/gm_data/' --save_path '../model14_output/' --max_words 80000 --class_weight