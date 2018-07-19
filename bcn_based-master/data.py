###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################


import os, helper, json
from nltk import pos_tag
from collections import Counter

pos_to_idx = {
    'CC': 0,
    'CD': 1,
    'DT': 2,
    'EX': 3,
    'FW': 4,
    'IN': 5,
    'JJ': 6,
    'JJR': 7,
    'JJS': 8,
    'LS': 9,
    'MD': 10,
    'NN': 11,
    'NNS': 12,
    'NNP': 13,
    'NNPS': 14,
    'PDT': 15,
    'POS': 16,
    'PRP': 17,
    'PRP$': 18,
    'RB': 19,
    'RBR': 20,
    'RBS': 21,
    'RP': 22,
    'SYM': 23,
    'TO': 24,
    'UH': 25,
    'VB': 26,
    'VBD': 27,
    'VBG': 28,
    'VBN': 29,
    'VBP': 30,
    'VBZ': 31,
    'WDT': 32,
    'WP': 33,
    'WP$': 34,
    'WRB': 35,
    'UNK': 36
}

class Dictionary(object):
    """Dictionary class that stores all words of train/dev corpus."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.pad_token = '<p>'
        self.idx2word.append(self.pad_token)
        self.word2idx[self.pad_token] = len(self.idx2word) - 1

    def build_dict(self, instances, max_words):
        word_count = Counter()
        for instance in instances:
            word_count.update(instance.sentence1)
            word_count.update(instance.sentence2)

        most_common = word_count.most_common(max_words) if max_words > 0 else word_count.most_common()
        for (index, w) in enumerate(most_common):
            self.idx2word.append(w[0])
            self.word2idx[w[0]] = len(self.idx2word) - 1

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Instance(object):
    """Instance that represent a sample of train/dev/test corpus."""

    def __init__(self):
        self.sentence1 = []
        self.sentence1_str = ""
        self.sentence2 = []
        self.label = -1
        self.pos_tags =  []


    def add_sentence(self, sentence, tokenize, sentence_no):
        words = ['<s>'] + helper.tokenize(sentence, tokenize) + ['</s>']
        self.sentence1_str = sentence
        if sentence_no == 1:
            self.sentence1 = words
        else:
            self.sentence2 = words

    def add_pos_tags(self, sentence, tokenize):
        """We assume that the raw sentence will be passed in -> this is for self-attentive network
        :param sentence: raw sentence not tokenized
        :return: void
        """
        tokenized_sent = helper.tokenize(sentence, tokenize)
        pos_tags = pos_tag(tokenized_sent)
        self.pos_tags = [tag[1] for tag in pos_tags]
        for i in range(len(self.pos_tags)):
            if self.pos_tags[i] not in pos_to_idx:
                self.pos_tags[i] = 'UNK'

    def add_label(self, label):
        self.label = label

    def print_(self):
        print (' sen 1: ', self.sentence1)
        print (' sen 2: ', self.sentence2)
        print (' label: ', self.label)



class Corpus(object):
    """Corpus class which contains all information about train/dev/test corpus."""

    def __init__(self, _tokenize):
        self.tokenize = _tokenize
        self.data = []
        self.class_distribution = {}

    def add_to_distribution(self, label_name):
        if label_name in self.class_distribution:
            self.class_distribution[label_name] += 1
        else:
            self.class_distribution[label_name] = 1

    def parse(self, in_file, task_name, max_example=None):
        """Parses the content of a file."""
        

        if 'IMDB' in task_name:
            for line in in_file:
                instance = Instance()
                instance.add_sentence(line['text'], self.tokenize, 1)
                instance.add_sentence(line['text'], self.tokenize, 2)
                instance.add_label(int(line['y']))
                self.data.append(instance)
                if len(self.data) == max_example:
                    break

        else:
            print("data.py parse:", in_file)
            assert os.path.exists(in_file)
            with open(in_file, 'r') as f:
                i = 0
                for line in f:
                    tokens = line.strip().split('\t')
                    instance = Instance()
                    if task_name != 'yelp':
                        instance.add_sentence(tokens[0], self.tokenize, 1)
                        instance.add_sentence(tokens[1], self.tokenize, 2)

                    if task_name == 'quora':
                        instance.add_label(int(tokens[2]))
                        self.add_to_distribution(int(tokens[2]))
                    elif task_name == 'snli' or task_name == 'multinli':
                        if tokens[4] == 'entailment':
                            instance.add_label(0)
                            self.add_to_distribution(label_name='entailment')
                        elif tokens[4] == 'neutral':
                            instance.add_label(1)
                            self.add_to_distribution(label_name='neutral')
                        elif tokens[4] == 'contradiction':
                            instance.add_label(2)
                            self.add_to_distribution(label_name='contradiction')
                    elif task_name == 'yelp':
                        #duplicate the inputs for biattentive model
                        instance.add_sentence(tokens[1], self.tokenize, 1)
                        instance.add_sentence(tokens[1], self.tokenize, 2)
#                        print(i)

                        instance.add_pos_tags(tokens[1], self.tokenize)
                        instance.add_label(int(tokens[0]))
                        self.add_to_distribution(label_name=int(tokens[0]))

                    self.data.append(instance)
                    if len(self.data) == max_example:
                        break
                    i+=1