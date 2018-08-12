#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python test.py --cuda --gpu 0 --task yelp --data '../data/yelp/gm_data/' --save_path '../model12_output/' --pos --class_weight