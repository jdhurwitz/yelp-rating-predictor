#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python main.py --cuda --gpu 0 --task yelp --data '../data/' --save_path '../model11_output/' --emtraining --class_weight