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


def decode(args, model, neg):
    inputs = build_all(neg, args.edit)
    in_size = inputs.size(1)
    try:
        if args.use_cuda:
            inputs = inputs.cuda(args.device, non_blocking=True)
        outputs = model(inputs)
        vals, idxs = torch.topk(outputs, args.topd, dim=1, largest=False)
        inputs_topd = inputs[0][idxs].squeeze(0)
        decodes = set()
        for i in range(args.topd):
            decode_i = inputs_topd[i]
            decode_i = ''.join([all_letters[j]
                                for j in decode_i.squeeze(0) if j != 0])
            decodes.add(decode_i)
        del inputs
        del outputs
        return in_size, decodes
    except:
        print("Inputs too big: {}".format(inputs.size(1)))
        return in_size, []


def test_decoder(args, model, vocab):
    model.eval()
    vocabset = set(vocab)  # faster lookup
    correct = 0
    total = 0
    tot_in_size = 0
    with torch.no_grad():
        for i, word in enumerate(vocab):
            neg = get_random_negative(list(word), vocabset, args.edit)
            if not neg:
                continue
            in_size, word_decodes = decode(args, model, neg)
            # the correct word is in the topk.
            if word in word_decodes:
                correct += 1
            tot_in_size += in_size
            total += 1
            if i % args.log_rate == 0:
                print("\rDecoded [{}/{}] ({:.0f}%) words, Acc: {:.3f}, avg_len: {:.0f}"
                      .format(i, args.topk, 100. * i/args.topk, correct/total, tot_in_size/total), end='')
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
    parser.add_argument('--topk', type=int, default=100000,
                        help="train on top k most common words in vocabulary")
    parser.add_argument('--topd', type=int, default=1,
                        help="evaluate success on top decode choices")
    parser.add_argument('--log_rate', type=float, default=1000,
                        help='number of samples per log (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # some settings for training on specific edits.
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
    vocab, _ = read_vocab(args.vocab_file, topk=args.topk)

    print("Evaluating on top {} words".format(args.topk))
    print("Using top {} decodes for each word".format(args.topd))
    start = time.time()
    test_decoder(args, model, vocab)
    print("Decoding time: {} sec".format(time.time()-start))


if __name__ == "__main__":
    main()
