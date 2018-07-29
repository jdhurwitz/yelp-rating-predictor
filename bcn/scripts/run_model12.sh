#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python main.py --cuda --gpu 0 --task yelp --data '../data/' --save_path '../model12_output/' --pos --class_weight