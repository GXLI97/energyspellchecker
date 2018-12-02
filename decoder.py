import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from loss import Energy_Loss
from utils import *
import numpy as np


def decode(model, line, min_len=7):
    # one positive and num_neg negative examples.
    idxs = [letterToIndex(l) for l in line]
    inputs = torch.tensor(idxs, dtype=torch.long).unsqueeze(0)
    for i in range(len(idxs)-1):
        swapped_idxs = idxs.copy()
        swapped_idxs[i], swapped_idxs[i+1] = swapped_idxs[i+1], swapped_idxs[i]
        temp = torch.tensor(swapped_idxs, dtype=torch.long).unsqueeze(0)
        inputs = torch.cat((inputs, temp), 0)
    if inputs.size(1) < min_len:
        pad = n_chars * torch.ones(inputs.size(0),
                                   min_len-inputs.size(1), dtype=torch.long)
        inputs = torch.cat((inputs, pad), 1)
    scores = model(inputs)
    v, i = scores.min(0)
    decode = ''.join([all_letters[j] for j in inputs[i].squeeze(0) if j != n_chars])
    return decode


def test(model, testset):
    model.eval()

    correct = 0
    total = 0

    # yes = []
    # no = []
    for i in range(1000):
        word = random.choice(testset)
        if len(word) < 4:
            continue
        r = random.randint(0, len(word)-2)
        temp = list(word)
        temp[r], temp[r+1] = temp[r+1], temp[r]
        word_bad = ''.join(temp)
        word_decode = decode(model, word_bad)
        if word_decode in testset:
            correct += 1
            # yes.append(freq_dict[word])
        # else:
            # print(word, word_bad, word_decode)
            # no.append(freq_dict[word])
        total += 1

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
    device = torch.device("cuda" if use_cuda else "cpu")
    # unclear what this is for...
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # instantiate CNN, loss, and optimizer.
    model = CNN(n_chars, 10, 1, 256, [1, 2, 3, 4, 5, 6, 7], 0, 1)
    model.load_state_dict(torch.load(args.model_save_file))
    model = model.to(device)
    vocab, freq_dict = read_vocab(args.vocab_file, topk=args.topk)
    
    test(model, vocab)

if __name__ == "__main__":
    main()




