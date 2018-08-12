#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python test.py --cuda --gpu 0 --task yelp --data '../data/yelp/gm_data/' --save_path '../model16_output/' --max_words 80000 --pos --class_weight