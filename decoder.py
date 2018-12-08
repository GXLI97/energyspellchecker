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


def decode(args, model, neg, topk):
    inputs = build_all(args, neg)
    if args.use_cuda:
            inputs = inputs.cuda(args.device, non_blocking=True)
    outputs = model(inputs)
    vals, idxs = torch.topk(outputs, topk, dim=1, largest=False)
    inputs_topk = inputs[0][idxs].squeeze(0)
    decodes = []
    for i in range(topk):
        decode_i = inputs_topk[i]
        decode_i = ''.join([all_letters[j]
                            for j in decode_i.squeeze(0) if j != 0])
        decodes.append(decode_i)
    return decodes


def test_decoder(args, model, vocab, topk=5):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, word in enumerate(vocab):
            neg = None
            while neg is None:
                neg = get_random_negative(list(word), vocab)
            word_decodes = decode(args, model, neg, topk)
            # at least one of topk words in vocab
            if [i for i in word_decodes if i in vocab]:
                correct += 1
            total += 1

            # correct word is amongst topk
            # if word in word_decodes:
                
            # else:
            #     print(word, word_decode)
            
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
    parser.add_argument('--topk', type=int, default=50000,
                        help="train on top k most common words in vocabulary")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    args = parser.parse_args()
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    print(args.device)

    # instantiate CNN, loss, and optimizer.
    model = CNN(n_chars, 10, 1, 256, [1, 2, 3, 4, 5, 6, 7, 8], 0.25, 1).to(device=args.device)
    # TODO: understand why this works...
    if args.use_cuda:
        model.load_state_dict(torch.load(args.model_save_file))
    else:
        model.load_state_dict(torch.load(args.model_save_file, map_location=lambda storage, loc: storage))
    vocab, freq_dict = read_vocab(args.vocab_file, topk=args.topk)
    start = time.time()
    test_decoder(args, model, vocab)
    print("Decoding time: {} sec".format(time.time()-start))


if __name__ == "__main__":
    main()
