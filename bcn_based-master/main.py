###############################################################################
# Author: Wasi Ahmad
# Adapted by: Jonathan Hurwitz
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script is the entry point of the entire pipeline.
###############################################################################


import util, helper, data, train, os, sys, numpy, torch, pickle, json
from torch import optim
from model import BCN

args = util.get_args()
# if output directory doesn't exist, create it
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# Set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

print('\ncommand-line params : {0}\n'.format(sys.argv[1:]))
print('{0}\n'.format(args))

###############################################################################
# Load data
###############################################################################

# load train and dev dataset
train_corpus = data.Corpus(args.tokenize)
dev_corpus = data.Corpus(args.tokenize)
test_corpus = data.Corpus(args.tokenize)

task_names = ['snli', 'multinli'] if args.task == 'allnli' else [args.task]
for task in task_names:
    if 'IMDB' in args.task:
        ###############################################################################
        # Load Learning to Skim paper's Pickle file
        ###############################################################################
        train_d, dev_d, test_d = helper.get_splited_imdb_data(args.save_path+'data/'+'imdb.p')
        train_corpus.parse(train_d, task, args.max_example)
        dev_corpus.parse(dev_d, task, args.max_example)
        test_corpus.parse(test_d, task, args.max_example)
    else:
        train_corpus.parse(args.data + task + '/train.txt', task, args.max_example)
        if task == 'multinli':
            dev_corpus.parse(args.data + task + '/dev_matched.txt', task, args.tokenize)
            test_corpus.parse(args.data + task + '/test_matched.txt', task, args.tokenize)
        else:
            dev_corpus.parse(args.data + task + '/dev.txt', task, args.tokenize)
            test_corpus.parse(args.data + task + '/test.txt', task, args.tokenize)


print('train set size = ', len(train_corpus.data))
print('development set size = ', len(dev_corpus.data))
print('test set size = ', len(test_corpus.data))

#pull the distributions
class_distributions = {}
class_distributions['train'] = train_corpus.class_distribution
#check to see if train set is actually imbalanced
if args.class_weight is True:
    if len(
            set(
                class_distributions['train'].values()
            )
    ) == 1: #we have all one value -> no need to weight
        args.class_weight = False


class_distributions['dev'] = dev_corpus.class_distribution
class_distributions['test'] = test_corpus.class_distribution

dictionary = data.Dictionary()
dictionary.build_dict(train_corpus.data + dev_corpus.data + test_corpus.data, args.max_words)
# save the dictionary object to use during testing
helper.save_object(dictionary, args.save_path + 'dictionary.p')
print('vocabulary size = ', len(dictionary))

embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file, dictionary.word2idx)
print('number of OOV words = ', len(dictionary) - len(embeddings_index))

# ###############################################################################
# # Build the model
# ###############################################################################

model = BCN(dictionary, embeddings_index, class_distributions, args)
print(model)
optim_fn, optim_params = helper.get_optimizer(args.optimizer)
optimizer = optim_fn(filter(lambda p: p.requires_grad, model.parameters()), **optim_params)
best_acc = 0

param_dict = helper.count_parameters(model)
print('number of trainable parameters = ', numpy.sum(list(param_dict.values())))

if args.cuda and torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    model = model.cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = helper.load_checkpoint(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# ###############################################################################
# # Train the model
# ###############################################################################

train = train.Train(model, optimizer, dictionary, args, best_acc)
train.train_epochs(train_corpus, dev_corpus, test_corpus, args.start_epoch, args.epochs)

