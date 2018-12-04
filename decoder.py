import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from loss import Energy_Loss
from utils import *
import numpy as np


def decode(args, model, neg, topk):
    inputs = buildall(neg)
    inputs = inputs.to(args.device)
    scores = model(inputs)
    vals, idxs = torch.topk(scores, topk, dim=0, largest=False)
    inputs_topk = inputs[idxs]
    decodes = []
    for i in range(topk):
        decode_i = inputs_topk[i]
        decode_i = ''.join([all_letters[j]
                            for j in decode_i.squeeze(0) if j != n_chars])
        decodes.append(decode_i)
    return decodes


def test_decoder(model, data, n=100, topk=10):
    model.eval()

    correct = 0
    total = 0

    # yes = []
    # no = []
    for i in range(n):
        word = random.choice(data)
        if len(word) < 4:
            continue
        neg = get_random_negative(list(word), data)
        word_decodes = decode(args, model, neg, topk)
        # at least one of topk words in vocab
        # if [i for i in word_decodes if i in data]:
        # correct word is amongst topk
        if word in word_decodes:
            correct += 1
        # else:
        #     print(word, word_decode)
        total += 1
        # yes.append(freq_dict[word])
        # else:
        # print(word, word_bad, word_decode)
        # no.append(freq_dict[word])
    print(correct/total)

    # import matplotlib.pyplot as plt
    # start = 10
    # end = 1000
    # bins=np.geomspace(start, end, num=100)
    # plt.figure(1)
    # plt.subplot(211)
    # plot1 = plt.hist(yes, bins=bins, alpha=0.5, color='blue')
    # plot2 = plt.hist(no, bins=bins, alpha=0.5, color='green')

    # plt.subplot(212)
    # accuracy = [x/y for x,y in zip(plot1[0]+1, plot1[0]+plot2[0]+1)]
    # diff=plt.plot(bins[:-1], accuracy,color='red')
    # plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Decoding spelling errors')
    parser.add_argument('--vocab_file', type=str, default="data/wikipedia.vocab.txt",
                        help='vocabulary file path')
    parser.add_argument('--model_save_file', type=str, default='models/energy_cnn',
                        help='model save path')
    parser.add_argument('--topk', type=int, default=10000,
                        help="train on top k most common words in vocabulary")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    print(args.device)

    # instantiate CNN, loss, and optimizer.
    model = CNN(n_chars, 10, 1, 256, [1, 2, 3, 4, 5, 6], 0.1, 1)
    model.load_state_dict(torch.load(args.model_save_file))
    model = model.to(device)
    vocab, freq_dict = read_vocab(args.vocab_file, topk=args.topk)

    test_decoder(args, model, vocab)


if __name__ == "__main__":
    main()
