###############################################################################
# Author: Wasi Ahmad
# Adapted by: Jonathan Hurwitz
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script contains code to train the model.
###############################################################################

import time, helper, torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, optimizer, dictionary, config, best_acc):
        self.model = model
        self.dictionary = dictionary
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        if self.config.cuda and torch.cuda.is_available():
            self.criterion = self.criterion.cuda()

        self.optimizer = optimizer
        self.best_dev_acc = best_acc
        self.times_no_improvement = 0
        self.stop = False
        self.train_accuracies = []
        self.dev_accuracies = []

    def train_epochs(self, train_corpus, dev_corpus, test_corpus, start_epoch, n_epochs):
        """Trains model for n_epochs epochs"""
        for epoch in range(start_epoch, start_epoch + n_epochs):
            if not self.stop:
                print('\nTRAINING : Epoch ' + str((epoch + 1)))
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * self.config.lr_decay \
                    if epoch > start_epoch and 'sgd' in self.config.optimizer else self.optimizer.param_groups[0]['lr']
                if 'sgd' in self.config.optimizer:
                    print('Learning rate : {0}'.format(self.optimizer.param_groups[0]['lr']))
                self.train(train_corpus)
                # training epoch completes, now do validation
                print('\nVALIDATING : Epoch ' + str((epoch + 1)))
                dev_acc = self.validate(dev_corpus)
                self.dev_accuracies.append(dev_acc)
                print('validation acc = %.2f%%' % dev_acc)
                test_acc = self.validate(test_corpus)
                print('validation acc = %.2f%%' % test_acc)
                # save model if dev accuracy goes up
                if self.best_dev_acc < dev_acc:
                    self.best_dev_acc = dev_acc
                    helper.save_checkpoint({
                        'epoch': (epoch + 1),
                        'state_dict': self.model.state_dict(),
                        'best_acc': self.best_dev_acc,
                        'optimizer': self.optimizer.state_dict(),
                    }, self.config.save_path + 'model_best.pth.tar')
                    self.times_no_improvement = 0
                else:
                    if 'sgd' in self.config.optimizer:
                        self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] / self.config.lrshrink
                        print('Shrinking lr by : {0}. New lr = {1}'.format(self.config.lrshrink,
                                                                           self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < self.config.minlr:
                            self.stop = True
                    if 'adam' in self.config.optimizer:
                        self.times_no_improvement += 1
                        # early stopping (at 'n'th decrease in accuracy)
                        if self.times_no_improvement == self.config.early_stop:
                            self.stop = True
                # save the train loss and development accuracy plot
                helper.save_plot(self.train_accuracies, self.config.save_path, 'training_acc_plot_', epoch + 1)
                helper.save_plot(self.dev_accuracies, self.config.save_path, 'dev_acc_plot_', epoch + 1)
            else:
                break

    def train(self, train_corpus):
        # Turn on training mode which enables dropout.
        self.model.train()

        # Splitting the data in batches
        train_batches = helper.batchify(train_corpus.data, self.config.batch_size)
        print('number of train batches = ', len(train_batches))

        start = time.time()
        print_acc_total = 0
        plot_acc_total = 0

        num_batches = len(train_batches)
        for batch_no in range(1, num_batches + 1):
            # Clearing out all previous gradient computations.
            self.optimizer.zero_grad()
            train_sentences1, sent_len1, train_sentences2, sent_len2, train_labels = helper.batch_to_tensors(
                train_batches[batch_no - 1], self.dictionary)
            if self.config.cuda and torch.cuda.is_available():
                train_sentences1 = train_sentences1.cuda()
                train_sentences2 = train_sentences2.cuda()
                train_labels = train_labels.cuda()

            assert train_sentences1.size(0) == train_sentences2.size(0)

            score = self.model(train_sentences1, sent_len1, train_sentences2, sent_len2)
            n_correct = (torch.max(score, 1)[1].view(train_labels.size()).data == train_labels.data).sum()
            # print (' score size ', score.size(), train_labels.size())
            loss = self.criterion(score, train_labels)
            # Important if we are using nn.DataParallel()
            if loss.size(0) > 1:
                loss = loss.mean()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            clip_grad_norm(filter(lambda p: p.requires_grad, self.model.parameters()), self.config.max_norm)
            self.optimizer.step()

            print_acc_total += 100. * n_correct / len(train_batches[batch_no - 1])
            plot_acc_total += 100. * n_correct / len(train_batches[batch_no - 1])

            if batch_no % self.config.print_every == 0:
                print_acc_avg = print_acc_total / self.config.print_every
                print_acc_total = 0
                print('%s (%d %d%%) %.2f' % (
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, print_acc_avg))

            if batch_no % self.config.plot_every == 0:
                plot_acc_avg = plot_acc_total / self.config.plot_every
                self.train_accuracies.append(plot_acc_avg)
                plot_acc_total = 0

    def validate(self, dev_corpus):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        dev_batches = helper.batchify(dev_corpus.data, self.config.batch_size)
        print('number of dev batches = ', len(dev_batches))

        num_batches = len(dev_batches)
        n_correct, n_total = 0, 0
        for batch_no in range(1, num_batches + 1):
            dev_sentences1, sent_len1, dev_sentences2, sent_len2, dev_labels = helper.batch_to_tensors(
                dev_batches[batch_no - 1], self.dictionary, True)
            if self.config.cuda and torch.cuda.is_available():
                dev_sentences1 = dev_sentences1.cuda()
                dev_sentences2 = dev_sentences2.cuda()
                dev_labels = dev_labels.cuda()

            assert dev_sentences1.size(0) == dev_sentences2.size(0)

            score = self.model(dev_sentences1, sent_len1, dev_sentences2, sent_len2)
            n_correct += (torch.max(score, 1)[1].view(dev_labels.size()).data == dev_labels.data).sum()
            n_total += len(dev_batches[batch_no - 1])

        return 100. * n_correct / n_total
