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


def get_edit(word, n_edits=1, edit=1):
    neg = []
    while not neg:
        neg = word.copy()
        for i in range(n_edits):
            # TODO: for now just use S W A P S, but later use this edit parameter.
            edits = [r_swap]
            neg = random.choice(edits)(neg)
    return neg


def beam_decode(args, model, neg):
    candidates = [neg]
    for i in range(args.n_edits):
        # print(candidates)
        examples = []
        for word in candidates:
            examples.append(word)
            for k in range(len(word)-1):
                ex = word.copy()
                ex[k], ex[k+1] = ex[k+1], ex[k]
                if ex not in examples:
                    examples.append(ex)
        inputs = torch.zeros(1, len(examples), max(
            len(word), min_len), dtype=torch.long)
        for i, ex in enumerate(examples, 0):
            vec_examples = [tok2index[tok] for tok in ex]
            inputs[0][i][:len(ex)] = torch.tensor(
                vec_examples, dtype=torch.long)
        # print(inputs)
        if args.use_cuda:
            inputs = inputs.cuda(args.device, non_blocking=True)
        outputs = model(inputs)
        # print(outputs)
        beam_size = min(args.beam_size, outputs.size(1))
        vals, idxs = torch.topk(outputs, beam_size, dim=1, largest=False)
        # print(vals)
        # print(idxs)
        inputs_top = inputs[0][idxs].squeeze(0)
        candidates = [[all_letters[j] for j in inputs_top[i] if j != 0]
                      for i in range(inputs_top.size(0))]
        # print(candidates)
    ret = "".join(candidates[0])
    # print(ret)
    # exit()
    return ret


def test_beam_decoder(args, model, vocab):
    model.eval()
    vocabset = set(vocab)
    correct1 = 0
    correct2 = 0
    total = 0
    with torch.no_grad():
        for i, word in enumerate(vocab):
            # compute a edit of this word
            neg = get_edit(list(word), n_edits=args.n_edits, edit=args.edit)
            # beam-decode it.
            word_decodes = beam_decode(args, model, neg)
            # print(word, "".join(neg), word_decodes)
            if word == word_decodes:
                correct1 += 1
            if word_decodes in vocabset:
                correct2 += 1
            total += 1
            if i % args.log_rate == 0:
                print("\rDecoded [{}/{}] ({:.0f}%) words, Acc: {:.3f} Acc: {:.3f}"
                      .format(i, args.topk, 100. * i/args.topk, correct1/total, correct2/total), end='')


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
    parser.add_argument('--beam_size', type=int, default=1,
                        help='how big to choose beam size.')
    # some settings for training on specific edits.
    parser.add_argument('--n_edits', type=int, default=1,
                        help='how many edits for each word.')
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

    print("testing beam decoding with numedits {} and beamsize {}".format(args.n_edits, args.beam_size))
    test_beam_decoder(args, model, vocab)


if __name__ == "__main__":
    main()
