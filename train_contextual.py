# imports
import argparse
import os
import itertools
import bcolz
import numpy as np
import pickle
from utils import *
import torch
from contextual_models import *

GLOVE_PATH = 'data'
EMB_DIM = 50


def build_glove_embeddings():
    # build embeddings
    words = []
    idx = 0
    word2idx = {}

    # restrict our vocabulary to the words that appear in text8.
    with open("data/text8", 'r') as f:
        text = [word for line in f for word in line.split()]
    textset = set(text)
    print(len(textset))
    if not os.path.isfile(f'{GLOVE_PATH}/6B.50.dat'):
        vectors = bcolz.carray(
            np.zeros(1), rootdir=f'{GLOVE_PATH}/6B.50.dat', mode='w')

        with open(f'{GLOVE_PATH}/glove.6B.50d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
        vectors = bcolz.carray(vectors[1:].reshape(
            (len(words), EMB_DIM)), rootdir=f'{GLOVE_PATH}/6B.50.dat', mode='w')
        vectors.flush()
        pickle.dump(words, open(f'{GLOVE_PATH}/6B.50_words.pkl', 'wb'))
        pickle.dump(word2idx, open(f'{GLOVE_PATH}/6B.50_idx.pkl', 'wb'))

    vectors = bcolz.open(f'{GLOVE_PATH}/6B.50.dat')[:]
    words = pickle.load(open(f'{GLOVE_PATH}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{GLOVE_PATH}/6B.50_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}

    print(words[:5])
    print("Glove embedding for the: {}".format(glove['the']))

    vocab = list(textset)
    vocab.insert(0, '<UNK>')

    # THIS IS THE ACTUAL LOOK UP TABLE, the previous one is not actually used...
    word2idx = {word: i for i, word in enumerate(vocab)}
    print("index of unk: {}".format(word2idx['<UNK>']))
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, EMB_DIM))

    words_found = 0
    for i, word in enumerate(vocab):
        try:
            weights_matrix[i, :] = glove[word]
            words_found += 1
        except KeyError:
            # print(word)
            if word == "<UNK>":
                weights_matrix[i, :] = np.zeros((50,))
            else:
                weights_matrix[i, :] = np.random.randn(50)*0.5
    weights_matrix = torch.from_numpy(weights_matrix).float()
    print(weights_matrix.size())
    print(weights_matrix[word2idx['the'], :])
    print(weights_matrix[word2idx['<UNK>'], :])
    return text, vocab, word2idx, weights_matrix


def train_cbow(args, model, text, vocab, word2idx):
    model.train()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(1):
        tot_loss = 0
        for i in range(2, len(text)-2):
            context = [text[i-2], text[i-1], text[i+1], text[i+2]]
            mid = text[i]
            context_idxs = torch.tensor(
                [word2idx[w] if w in word2idx else 0 for w in context], dtype=torch.long)
            mid_idx = torch.tensor(
                [word2idx[mid] if mid in word2idx else 0], dtype=torch.long)
            if args.use_cuda:
                context_idxs, mid_idx = context_idxs.cuda(args.device, non_blocking=True), mid_idx.cuda(args.device, non_blocking=True)
            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_function(log_probs, mid_idx)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            if i % 10 == 0:
                print('\r {:.2f}'.format(tot_loss/(i+1)), end='')
            if i == 100:
                break

    model.eval()
    # test the model here.
    for i in range(2, len(text)-2):
        context = [text[i-2], text[i-1], text[i+1], text[i+2]]
        mid = text[i]
        context_idxs = torch.tensor(
            [word2idx[w] if w in word2idx else 0 for w in context], dtype=torch.long)
        mid_idx = torch.tensor(
            [word2idx[mid] if mid in word2idx else 0], dtype=torch.long)
        if args.use_cuda:
            context_idxs, mid_idx = context_idxs.cuda(args.device, non_blocking=True), mid_idx.cuda(args.device, non_blocking=True)
        log_probs = model(context_idxs)
        print(log_probs.size())
        val, idx = torch.max(log_probs, 1)
        print(mid_idx)
        print(idx)
        tot_loss += loss.item()
        if i % 10 == 0:
            print('\r {:.2f}'.format(tot_loss/(i+1)), end='')
            break

def main():
    parser = argparse.ArgumentParser(
        description='Contextual Spell Correction')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    
    # read vocab.
    text, vocab, word2idx, weights_matrix = build_glove_embeddings()
    model = CBOW(weights_matrix, 4).to(device=args.device)
    train_cbow(args, model, text, vocab, word2idx)
    # initiate model

    # train model.

    pass


if __name__ == "__main__":
    main()
