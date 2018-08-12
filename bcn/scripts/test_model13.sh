#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python test.py --cuda --gpu 0 --task yelp --data '../data/yelp/gm_data/' --save_path '../model13_output/' --pos --emtraining --class_weight