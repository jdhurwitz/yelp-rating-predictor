#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py --cuda --gpu 0 --task yelp --data '../data/' --save_path '../model10_output/' --class_weight