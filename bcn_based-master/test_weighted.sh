#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py --cuda --gpu 0 --task yelp --save_path '../bcn_output_weighted/' --emtraining
