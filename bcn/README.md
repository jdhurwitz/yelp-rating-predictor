# Bi-Attentive Classification Network

This code was adapted from code written by Wasi Ahmad. Some of the code (encoder) is from Facebook research. Due to the network's size, you absolutely need to train on a GPU. You'll probably die before this finishes training if you try to do it on a CPU.

### Training the classifier 
-You can run the training script as follows:
```CUDA_VISIBLE_DEVICES=0 python main.py --cuda --gpu 0 --task yelp --data '<your data dir>' --save_path '<your save path>' ```
-If you have multiple GPUs, you can set which one to use by ```CUDA_VISIBLE_DEVICES```. This is zero-indexed, so 0 represents your first GPU. 
-You can enable word embedding training by ```--emtraining```.
-You can limit vocab size by ```--max_words <size integer>```.
-You can train with part of speech features: ```--pos```.

-A full list of flags can be found in ```util.py```



