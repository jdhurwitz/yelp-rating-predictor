###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script tests classification accuracy.
###############################################################################

import torch, helper, util, os, data
import np as np
from model import BCN
from sklearn.metrics import f1_score, classification_report


args = util.get_args()
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def evaluate(model, batches, dictionary, outfile=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    n_correct, n_total = 0, 0
    y_preds, y_true, output = [], [], []
    for batch_no in range(len(batches)):
        test_sentences1, sent_len1, test_sentences2, sent_len2, test_labels = helper.batch_to_tensors(batches[batch_no],
                                                                                                      dictionary, True)
        if args.cuda and torch.cuda.is_available():
            test_sentences1 = test_sentences1.cuda()
            test_sentences2 = test_sentences2.cuda()
            test_labels = test_labels.cuda()
        assert test_sentences1.size(0) == test_sentences1.size(0)

        score = model(test_sentences1, sent_len1, test_sentences2, sent_len2)
        preds = torch.max(score, 1)[1]
        if outfile:
            predictions = preds.data.cpu().tolist()
            for i in range(len(batches[batch_no])):
                output.append([batches[batch_no][i].id, predictions[i]])
        else:
            y_preds.extend(preds.data.cpu().tolist())
            y_true.extend(test_labels.data.cpu().tolist())
            p = preds.view(test_labels.size()).data
            n_correct += (preds.view(test_labels.size()).data == test_labels.data).sum()
            #n_same = (np.array(y_preds) == np.array(y_true)).sum()
#            n_correct += n_same
            n_total += len(batches[batch_no])

    clf_report = classification_report(np.array(y_true), np.array(y_preds))
    if outfile:
        target_names = ['entailment', 'neutral', 'contradiction']
        with open(outfile, 'w') as f:
            f.write('pairID,gold_label' + '\n')
            for item in output:
                f.write(str(item[0]) + ',' + target_names[item[1]] + '\n')
    else:
        return 100. * n_correct / n_total, 100. * f1_score(np.asarray(y_true), np.asarray(y_preds),
                                                           average='weighted'), clf_report


if __name__ == "__main__":
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    class_distributions=None
    model = BCN(dictionary, embeddings_index, class_distributions, args)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    helper.load_model_states_from_checkpoint(model, args.save_path + 'model_best.pth.tar', 'state_dict', args.cuda)
    print('vocabulary size = ', len(dictionary))

    task_names = ['snli', 'multinli'] if args.task == 'allnli' else [args.task]
    for task in task_names:
        test_corpus = data.Corpus(args.tokenize)
        if 'imdb' in args.task:
            ###############################################################################
            # Load Learning to Skim paper's Pickle file
            ###############################################################################
            train_d, dev_d, test_d = helper.get_splited_imdb_data(args.save_path + 'data/' + 'imdb.p')
            test_corpus.parse(test_d, task, args.max_example)

        elif task == 'multinli' and args.test != 'train':
            for partition in ['_matched', '_mismatched']:
                test_corpus.parse(args.data + task + '/' + args.test + partition + '.txt', task, args.max_example)
                print('[' + partition[1:] + '] dataset size = ', len(test_corpus.data))
                test_batches = helper.batchify(test_corpus.data, args.batch_size)
                if args.test == 'test':
                    evaluate(model, test_batches, dictionary, args.save_path + args.task + partition + '.csv')
                else:
                    test_accuracy, test_f1 = evaluate(model, test_batches, dictionary)
                    print('[' + partition[1:] + '] accuracy: %.2f%%' % test_accuracy)
                    print('[' + partition[1:] + '] f1: %.2f%%' % test_f1)
        else:
            test_corpus.parse(args.data + task + '/' + args.test + '.txt', task, args.max_example)
        print('dataset size = ', len(test_corpus.data))
        test_batches = helper.batchify(test_corpus.data, args.batch_size)
        test_accuracy, test_f1, classification_report = evaluate(model, test_batches, dictionary)
        print('accuracy: %.2f%%' % test_accuracy)
        print('f1: %.2f%%' % test_f1)
