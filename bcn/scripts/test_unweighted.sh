#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python test.py --cuda --gpu 0 --task yelp --save_path '../bcn_output_unweighted/' --emtraining
