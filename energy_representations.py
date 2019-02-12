import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from loss import Energy_Loss
from utils import *
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def build_batch(examples):
    inputs = torch.zeros(1, len(examples), max(len(examples[0])+5, min_len), dtype=torch.long)
    for i, ex in enumerate(examples, 0):
        vec_examples = [tok2index[tok] for tok in ex]
        inputs[0][i][:len(ex)] = torch.tensor(
            vec_examples, dtype=torch.long)
    return inputs


def plot_margins(args, model, vocab):
    model.eval()
    vocabset = set(vocab)
    with torch.no_grad():
        for edit in [1, 2, 3, 4]:
            margin_matrix = torch.zeros(args.num_samples, args.max_k+1)
            for i in range(args.num_samples):
                examples = []
                word = list(random.choice(vocab))
                # compute the k-edit.
                examples.append(word)
                for k in range(1, args.max_k+1):
                    neg = get_edit(word, k, edit)
                    if "".join(neg) in vocabset:
                        examples.append([])
                    else:
                        examples.append(neg)
                inputs = build_batch(examples)
                if args.use_cuda:
                    inputs = inputs.cuda(args.device, non_blocking=True)
                outputs, _ = model(inputs)
                margin = outputs - outputs[0][0]
                margin_matrix[i] = margin
            
            avgs = [0]
            lbs = [0]
            ubs = [0]
            for col in range(1, args.max_k+1):
                energies = margin_matrix[:,col]
                avgs.append(torch.mean(energies).item())
                lbs.append(np.percentile(energies, 5))
                ubs.append(np.percentile(energies, 95))
            print(avgs)
            print(lbs)
            print(ubs)
            avgs = np.array(avgs)
            lbs = np.array(lbs)
            ubs = np.array(ubs)

            plt.figure() 
            plt.errorbar(range(0, args.max_k+1), avgs, yerr=[avgs-lbs, ubs-avgs], fmt='-s')
            plt.savefig('edit{}_margin.png'.format(edit))

def type_edit_pca(args, model, vocab):
    model.eval()
    vocabset = set(vocab)
    with torch.no_grad():
        data_matrix = torch.zeros(args.num_samples, 5, 1000)
        labels = torch.zeros(args.num_samples, 5)
        for i in range(args.num_samples):
            examples = []
            word = list(random.choice(vocab))
            # compute the k-edit.
            examples.append(word)
            for edit in [1, 2, 3, 4]:
                neg = get_edit(word, 1, edit)
                if "".join(neg) in vocabset:
                    examples.append([])
                else:
                    examples.append(neg)
            inputs = build_batch(examples)
            if args.use_cuda:
                inputs = inputs.cuda(args.device, non_blocking=True)
            outputs, reps = model(inputs)
            data_matrix[i] = reps
            labels[i] = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        data_matrix = data_matrix.view(args.num_samples*5,-1)
        labels = labels.view(args.num_samples*5,-1)
        print(data_matrix.size())
        print(labels.numpy().reshape((args.num_samples*5,)).tolist())
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_matrix)
        print(data_pca.shape)
        plt.scatter(data_pca[:,0], data_pca[:,1],s=1, alpha=0.75,
                    c=labels.numpy().reshape((args.num_samples*5,)).tolist(), cmap=plt.cm.get_cmap('Set1', 5))
        plt.colorbar()
        # plt.show()
        plt.savefig("edit_type_pca.png")
        


def k_edit_pca(args, model, vocab):
    model.eval()
    vocabset = set(vocab)
    with torch.no_grad():
        data_matrix = torch.zeros(args.num_samples, 5, 1000)
        labels = torch.zeros(args.num_samples, 5)
        for i in range(args.num_samples):
            examples = []
            word = list(random.choice(vocab))
            # compute the k-edit.
            examples.append(word)
            for k in range(1, args.max_k+1):
                neg = get_edit(word, k, 4)
                if "".join(neg) in vocabset:
                    examples.append([])
                else:
                    examples.append(neg)
            inputs = build_batch(examples)
            if args.use_cuda:
                inputs = inputs.cuda(args.device, non_blocking=True)
            outputs, reps = model(inputs)
            data_matrix[i] = reps
            labels[i] = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        data_matrix = data_matrix.view(args.num_samples*5,-1)
        labels = labels.view(args.num_samples*5,-1)
        print(data_matrix.size())
        print(labels.numpy().reshape((args.num_samples*5,)).tolist())
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_matrix)
        print(data_pca.shape)
        plt.scatter(data_pca[:,0], data_pca[:,1],s=1, alpha=0.75,
                    c=labels.numpy().reshape((args.num_samples*5,)).tolist(), cmap=plt.cm.get_cmap('Set1', 5))
        plt.colorbar()
        plt.savefig("rep_k_pca.png")


    


def main():
    parser = argparse.ArgumentParser(
        description='Visualizing margins for energy model')
    parser.add_argument('--vocab_file', type=str, default="data/wikipedia.vocab.txt",
                        help='vocabulary file path')
    parser.add_argument('--model_save_file', type=str, default='models/energy_cnn',
                        help='model save path')
    parser.add_argument('--topk', type=int, default=100000,
                        help="train on top k most common words in vocabulary")
    parser.add_argument('--max_k', type=int, default=4,
                        help="maximum number of edits 0-k")
    parser.add_argument('--num_samples', type=int, default=10000,
                        help="how many samples for each edit value")
    parser.add_argument('--log_rate', type=float, default=1000,
                        help='number of samples per log (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    print("Using Device: {}".format(args.device))

    # instantiate CNN, loss, and optimizer.
    print("Loading Model from {}".format(args.model_save_file))
    model = CNN(n_chars, 10, 1, 256, [1, 2, 3, 4, 5, 6, 7, 8], 0.25, 1000, 1).to(
        device=args.device)
    # some weird loading, TODO: test this out on gpu.
    if args.use_cuda:
        model.load_state_dict(torch.load(args.model_save_file))
    else:
        model.load_state_dict(torch.load(
            args.model_save_file, map_location=lambda storage, loc: storage))

    # load vocabulary.
    vocab, _ = read_vocab(args.vocab_file, topk=args.topk)

    # plot_margins(args, model, vocab)
    # type_edit_pca(args, model, vocab)
    k_edit_pca(args, model, vocab)

if __name__ == "__main__":
    main()