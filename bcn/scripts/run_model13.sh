#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python main.py --cuda --gpu 0 --task yelp --data '../data/' --save_path '../model13_output/' --pos --emtraining --class_weight