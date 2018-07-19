###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script provides general purpose utility functions that
# may come in handy at any point in the experiments.
###############################################################################

import re, os, glob, pickle, inspect, math, time, torch, util, json
import numpy as np
from torch import optim
from nltk import word_tokenize
from collections import OrderedDict
from torch.autograd import Variable
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#get the pos dictionary
from data import pos_to_idx

args = util.get_args()


def load_word_embeddings(directory, file, dictionary):
    embeddings_index = {}
    f = open(os.path.join(directory, file))
    for line in f:
        word, vec = line.split(' ', 1)
        if word in dictionary:
            embeddings_index[word] = np.array(list(map(float, vec.split())))
    f.close()
    return embeddings_index


def save_word_embeddings(directory, file, embeddings_index):
    f = open(os.path.join(directory, file), 'w')
    for word, vec in embeddings_index.items():
        f.write(word + ' ' + ' '.join(str(x) for x in vec) + '\n')
    f.close()


def load_checkpoint(filename, from_gpu=True):
    """Load a previously saved checkpoint."""
    assert os.path.exists(filename)
    if from_gpu:
        return torch.load(filename)
    else:
        return torch.load(filename, map_location=lambda storage, loc: storage)


def save_checkpoint(state, filename='./checkpoint.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)


def get_optimizer(s, verbose = 0):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if verbose: 
        print (' method: ', method, ' optim_params: ', optim_params)

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = list(inspect.signature(optim_fn.__init__).parameters.keys())
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


def load_model_states_from_checkpoint(model, filename, tag, from_gpu=True):
    """Load model states from a previously saved checkpoint."""
    assert os.path.exists(filename)
    if from_gpu:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint[tag])


def load_model_states_without_dataparallel(model, filename, tag):
    """Load a previously saved model states."""
    assert os.path.exists(filename)
    checkpoint = torch.load(filename)
    new_state_dict = OrderedDict()
    for k, v in checkpoint[tag].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def save_object(obj, filename):
    """Save an object into file."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_object(filename):
    """Load object from file."""
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def count_parameters(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dict[name] = np.prod(param.size())
    return param_dict


def tokenize(s, tokenize):
    """Tokenize string."""
    if tokenize:
        return word_tokenize(s)
    else:
        return s.split()


def initialize_out_of_vocab_words(dimension, choice='zero'):
    """Returns a vector of size dimension given a specific choice."""
    if choice == 'random':
        """Returns a random vector of size dimension where mean is 0 and standard deviation is 1."""
        return np.random.normal(size=dimension)
    elif choice == 'zero':
        """Returns a vector of zeros of size dimension."""
        return np.zeros(shape=dimension)


def batchify(data, bsz):
    """Transform data into batches."""
    np.random.shuffle(data)
    batched_data = []
    for i in range(len(data)):
        if i % bsz == 0:
            batched_data.append([data[i]])
        else:
            batched_data[len(batched_data) - 1].append(data[i])
    return batched_data


def convert_to_minutes(s):
    """Converts seconds to minutes and seconds"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def show_progress(since, percent):
    """Prints time elapsed and estimated time remaining given the current time and progress in %"""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (convert_to_minutes(s), convert_to_minutes(rs))


def save_plot(points, filepath, filetag, epoch):
    """Generate and save the plot"""
    path_prefix = os.path.join(filepath, filetag)
    path = path_prefix + 'epoch_{}.png'.format(epoch)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.plot(points)
    fig.savefig(path)
    plt.close(fig)  # close the figure
    for f in glob.glob(path_prefix + '*'):
        if f != path:
            os.remove(f)


def show_plot(points):
    """Generates plots"""
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def sequence_to_tensor(sequence, max_sent_length, dictionary):
    """Convert a sequence of words to a tensor of word indices."""
    sen_rep = torch.LongTensor(max_sent_length).zero_()
    for i in range(len(sequence)):
        if dictionary.contains(sequence[i]):
            sen_rep[i] = dictionary.word2idx[sequence[i]]
    return sen_rep

def pos_to_tensor(sequence, max_sent_length, dictionary):
    tensor_rep = torch.LongTensor(max_sent_length).zero_()
    for i in range(len(sequence)):
        if sequence[i] in pos_to_idx:
            tensor_rep[i] = pos_to_idx[sequence[i]]
    return tensor_rep

def batch_to_tensors(batch, dictionary, iseval=False, pos=False):
    """Convert a list of sequences to a list of tensors."""
    max_sent_length1, max_sent_length2 = 0, 0
    for item in batch:
        if max_sent_length1 < len(item.sentence1):
            max_sent_length1 = len(item.sentence1)
        if max_sent_length2 < len(item.sentence2):
            max_sent_length2 = len(item.sentence2)

    all_sentences1 = torch.LongTensor(len(batch), max_sent_length1)
    sent_len1 = np.zeros(len(batch), dtype=np.int)
    all_sentences2 = torch.LongTensor(len(batch), max_sent_length2)
    sent_len2 = np.zeros(len(batch), dtype=np.int)


    all_pos1 = torch.LongTensor(len(batch), max_sent_length1)
    all_pos2 = torch.LongTensor(len(batch), max_sent_length2)

    labels = torch.LongTensor(len(batch))
    for i in range(len(batch)):
        sent_len1[i], sent_len2[i] = len(batch[i].sentence1), len(batch[i].sentence2)
        all_sentences1[i] = sequence_to_tensor(batch[i].sentence1, max_sent_length1, dictionary)
        all_sentences2[i] = sequence_to_tensor(batch[i].sentence2, max_sent_length2, dictionary)

        if pos: #conditional for efficiency
            all_pos1[i] = pos_to_tensor(batch[i].pos_tags, max_sent_length1, pos_to_idx)
            all_pos2[i] = pos_to_tensor(batch[i].pos_tags, max_sent_length2, pos_to_idx)

        labels[i] = batch[i].label

    if iseval:
        return Variable(all_sentences1, volatile=True), sent_len1, Variable(all_sentences2,volatile=True), sent_len2, \
                   Variable(labels, volatile=True), Variable(all_pos1, volatile=True), Variable(all_pos2, volatile=True)

    else:
        return Variable(all_sentences1), sent_len1, Variable(all_sentences2), sent_len2, Variable(labels), Variable(all_pos1), Variable(all_pos2)



def get_splited_imdb_data(file_name, data_name='IMDB'):
    if data_name=='IMDB':
        train_d =[]
        dev_d =[]
        test_d = []
        with open(file_name, 'rb') as f:
            x = pickle.load(f, encoding="latin1")
            all_d = x[0]
            for line in all_d:
                if line['split']==0: train_d.append(line)
                elif line['split']==1: dev_d.append(line)
                else: test_d.append(line)
        return train_d, dev_d, test_d
