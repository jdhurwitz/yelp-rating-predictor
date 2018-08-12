#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python test.py --cuda --gpu 0 --task yelp --data '../data/yelp/gm_data/' --save_path '../model10_output/' --class_weight