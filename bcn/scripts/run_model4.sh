#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python main.py --cuda --gpu 0 --task yelp --data '../../data/yelp/gm_data/' --save_path '../../model4_output/' --pos --emtraining