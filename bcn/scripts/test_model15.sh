#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python test.py --cuda --gpu 0 --task yelp --data '../data/yelp/gm_data/' --save_path '../model15_output/' --max_words 80000 --emtraining --class_weight