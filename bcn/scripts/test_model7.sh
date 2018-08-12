#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python test.py --cuda --gpu 0 --task yelp --data '../data/' --save_path '../model7_output/' --max_words 80000 --pos