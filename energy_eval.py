import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from loss import Energy_Loss
from torch.utils.data import DataLoader
from utils import *

def energy_eval(args, model, vocab, n=1000):
    # A LOT OF REDUNDANT CODE :(
    model.eval()
    vocabset = set(vocab)
    with torch.no_grad():
        e0 = []
        for i in range(n):
            word = random.choice(vocab)
            word = list(word)
            vec_word = [tok2index[tok] for tok in word]
            inputs = torch.zeros(1, 1, max(len(word), min_len), dtype=torch.long)
            inputs[0][0][:len(word)] = torch.tensor(vec_word, dtype=torch.long)
            if args.use_cuda:
                inputs = inputs.cuda(args.device, non_blocking=True)
            outputs = model(inputs)
            e0.append(outputs.cpu().numpy()[0][0])
        
        e1 = []
        for i in range(n):
            word = random.choice(vocab)
            word = list(word)
            neg = []
            while not neg:
                neg = get_random_negative(word, vocabset)
            vec_word = [tok2index[tok] for tok in neg]
            inputs = torch.zeros(1, 1, max(len(neg), min_len), dtype=torch.long)
            inputs[0][0][:len(neg)] = torch.tensor(vec_word, dtype=torch.long)
            if args.use_cuda:
                inputs = inputs.cuda(args.device, non_blocking=True)
            outputs = model(inputs)
            e1.append(outputs.cpu().numpy()[0][0])

        e2 = []
        for i in range(n):
            word = random.choice(vocab)
            word = list(word)
            neg = []
            while not neg:
                neg = get_random_negative(word, vocabset)
                neg = get_random_negative(neg, vocabset)
            vec_word = [tok2index[tok] for tok in neg]
            inputs = torch.zeros(1, 1, max(len(neg), min_len), dtype=torch.long)
            inputs[0][0][:len(neg)] = torch.tensor(vec_word, dtype=torch.long)
            if args.use_cuda:
                inputs = inputs.cuda(args.device, non_blocking=True)
            outputs = model(inputs)
            e2.append(outputs.cpu().numpy()[0][0])
        
        e3 = []
        for i in range(n):
            word = random.choice(vocab)
            word = list(word)
            neg = []
            while not neg:
                neg = get_random_negative(word, vocabset)
                neg = get_random_negative(neg, vocabset)
                neg = get_random_negative(neg, vocabset)
            vec_word = [tok2index[tok] for tok in neg]
            inputs = torch.zeros(1, 1, max(len(neg), min_len), dtype=torch.long)
            inputs[0][0][:len(neg)] = torch.tensor(vec_word, dtype=torch.long)
            if args.use_cuda:
                inputs = inputs.cuda(args.device, non_blocking=True)
            outputs = model(inputs)
            e3.append(outputs.cpu().numpy()[0][0])
        
    print(e0)
    print(e1)
    print(e2)
    print(e3)

    from matplotlib import pyplot as plt
    bins = np.linspace(-10,10,100)
    plt.hist(e0, alpha=0.5, label='e0')
    plt.hist(e1, alpha=0.5, label='e1')
    plt.hist(e2, alpha=0.5, label='e2')
    plt.hist(e3, alpha=0.5, label='e3')
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig('plt.png')

    # pick random word
    # build 1edit
    # build 2edit
    # build 3edit
    
    # save data
    # graph data.

def main():
    parser = argparse.ArgumentParser(
        description='Decoding spelling errors')
    parser.add_argument('--vocab_file', type=str, default="data/wikipedia.vocab.txt",
                        help='vocabulary file path')
    parser.add_argument('--model_save_file', type=str, default='models/energy_cnn',
                        help='model save path')
    parser.add_argument('--topk', type=int, default=100000,
                        help="train on top k most common words in vocabulary")
    parser.add_argument('--log_rate', type=float, default=1000,
                        help='number of samples per log (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # some settings for choosing specific edits.
    parser.add_argument('--edit', type=int, default=0,
                        help='choose a specific edit. 0:all, 1:swap, 2:add/del, 3:replace')
    args = parser.parse_args()

    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    print("Using Device: {}".format(args.device))

    # instantiate CNN, loss, and optimizer.
    print("Loading Model from {}".format(args.model_save_file))
    model = CNN(n_chars, 10, 1, 256, [1, 2, 3, 4, 5, 6, 7, 8], 0.25, 1).to(
        device=args.device)
    # some weird loading, TODO: test this out on gpu.
    if args.use_cuda:
        model.load_state_dict(torch.load(args.model_save_file))
    else:
        model.load_state_dict(torch.load(
            args.model_save_file, map_location=lambda storage, loc: storage))

    # load vocabulary.
    vocab, freq_dict = read_vocab(args.vocab_file, topk=args.topk)

    # energy_evaluator
    energy_eval(args, model, vocab)


if __name__ == "__main__":
    main()