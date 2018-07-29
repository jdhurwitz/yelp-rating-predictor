#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python main.py --cuda --gpu 0 --task yelp --data '../data/yelp/gm_data/' --save_path '../model3_output/' --pos 