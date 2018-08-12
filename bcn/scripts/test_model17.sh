#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py --cuda --gpu 0 --task yelp --data '../data/yelp/gm_data/' --save_path '../model17_output/' --max_words 80000 --emtraining --pos --class_weight