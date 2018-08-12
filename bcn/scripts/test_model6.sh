#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python test.py --cuda --gpu 0 --task yelp --data '../data/' --save_path '../model6_output/' --max_words 80000 --emtraining