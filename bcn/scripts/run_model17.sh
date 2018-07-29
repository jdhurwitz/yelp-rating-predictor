#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python main.py --cuda --gpu 0 --task yelp --data '../data/' --save_path '../model17_output/' --max_words 80000 --emtraining --pos --class_weight