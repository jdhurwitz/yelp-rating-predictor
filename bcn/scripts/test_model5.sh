#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python test.py --cuda --gpu 0 --task yelp --data '../data/' --save_path '../model5_output/' --max_words 80000