Yelp Rating Predictor 
======
The bulk of the work exists in the 'bcn' directory rather than in any of the jupyter notebook files. 

### What does it do?
The self-attentive classification network learns a bijection between yelp text sentiment and a star rating. 
You can train your own model and then use it to suggest a star rating based on text. You can use your model's learned weights
in a transfer-learning approach for other sentiment analysis tasks as well.

### Do I need a GPU?
Yes. Since this is a deep learning model, you'll need to train it on a GPU. During development I used a GTX 1080 with 11GB of VRAM.
Don't even attempt to train it on a CPU. 

### How do I train it?
Make sure that you have the yelp dataset in your 'data' folder. If you have a different directory structure, you'll need to specify this via the ```--data``` flag in the training script call below. You can grab the full balanced dataset here:
https://drive.google.com/drive/u/1/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

After cloning the repo, navigate to the bcn folder. Here's an example of how to run the training script:
```CUDA_VISIBLE_DEVICES=0 python main.py --cuda --gpu 0 --task yelp```

If you're spoiled enough to have multiple GPUs, CUDA is zero-indexed so you'll want to tag which GPU to train on. Setting CUDA_VISIBLE_DEVICES to 1 would be using the second GPU. You can leave the ```--gpu``` flag as 0. There are a ton of different configurations you can set up. Check out util.py in order to see how you can tweak and customize the model during training. Here are some examples:

- **Changing the word embedding size**: You can select your own embedding size. The default is 300d.
  - `--emsize 150`

- **Training with custom embeddings**: You can train your own word embeddings.
  - `--emtraining`
   
- **Limiting the vocab size**: If you have a large vocab, it might make sense to reduce it to the most common x% of words. For my tests, I reduced the vocab to the 80,000 most frequent words. You can select your own number via the following flag: 
  - `--max_words <vocab size you want>`
  
- **Limiting the number of training examples**: If you want to speed up training and don't feel the need to use the whole dataset, you can limit the number of training examples:
  - `--max_example <your desired number>`
  
- **Number of training epochs**: You can train as long as you want, but when validation accuracy begins to decrease, the model will automatically stop and save the best performing model thus far. The default is 14 epochs.
  - `--epochs <number>`
  
- **Loss weighting**: If you have a class imbalance, you can penalize the majority class(es) during training via loss weighting. This uses an inverse probability in order to scale down the weights during backpropagation. To use it, add the following flag:
  - `--class_weight`
  
- **Part of speech tags as a feature**: Turning this flag on will use the NLTK tagger to tag parts of speech, which are then used as a feature. POS embedding dimension is 50 by default and has no configuration option, so you'll have to change that manually. To turn on POS tags:
  - `--pos`
  
- **Limiting the size of all sets for debugging**: If everything is broken and you need to debug locally, set the limit flag in order to limit the size of all sets to the number specified by max_example. This flag requires you to add the `--max_example <num>` flag.
  - `--limit`
   
Here's an example training script call with some custom flags:
```CUDA_VISIBLE_DEVICES=0 python main.py --cuda --gpu 0 --task yelp --emtraining --pos --max_words 80000```

There are many more options shown in util.py, including optimizer settings, layer numbers, unit counts for LSTM, and more. Please take a look at this file for full configuration options.

### How do I test it?
You can run the testing script as follows:
```CUDA_VISIBLE_DEVICES=0 python test.py --cuda --gpu 0 --task yelp```
