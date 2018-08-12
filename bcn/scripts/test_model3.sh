#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python test.py --cuda --gpu 0 --task yelp --data '../data/' --save_path '../model3_output/' --pos 