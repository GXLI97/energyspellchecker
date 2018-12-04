import string
import random
import math
import torch

all_letters = string.ascii_lowercase
n_chars = len(all_letters)
min_len = 8


def read_vocab(filename, topk=10000):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    lines = [line.split() for line in lines][:topk]
    random.shuffle(lines)
    vocab = [line[0] for line in lines]
    freq_dict = {line[0]: int(line[1]) for line in lines}
    return vocab, freq_dict


def letterToIndex(letter):
    return all_letters.find(letter)


def traintest_split(data, p=0.8):
    idx = math.floor(len(data)*p)
    trainset = data[:idx]
    testset = data[idx:]
    return trainset, testset


def r_swap(word):
    k = random.randint(0, len(word)-2)
    word[k], word[k+1] = word[k+1], word[k]
    return word


def r_add(word):
    k = random.randint(0, len(word))
    let = random.choice(string.ascii_lowercase)
    word.insert(k, let)
    return word


def r_del(word):
    k = random.randint(0, len(word)-1)
    del word[k]
    return word


def r_replace(word):
    k = random.randint(0, len(word)-1)
    let = random.choice(string.ascii_lowercase)
    word[k] = let
    return word


def get_random_negative(word, vocab):
    neg = None
    while neg is None or "".join(neg) in vocab:
        neg = word.copy()
        edits = [r_swap, r_add, r_del, r_replace]
        neg = random.choice(edits)(neg)
    return neg


def minibatch(args, word, vocab, num_neg):
    examples = []
    word = list(word)
    examples.append(word)
    for i in range(num_neg):
        examples.append(get_random_negative(word, vocab))
    inputs = torch.empty(len(examples), max(
        len(word)+1, min_len), device=args.device, dtype=torch.long)
    inputs[:] = n_chars
    for i, example in enumerate(examples, 0):
        idxs = [letterToIndex(l) for l in example]
        inputs[i][:len(example)] = torch.tensor(idxs, device=args.device, dtype=torch.long)
    labels = torch.zeros(len(examples), device=args.device, dtype=torch.long)
    labels[0] = 1
    return inputs, labels


def buildall(args, neg):
    examples = []
    neg = list(neg)
    examples.append(neg)
    # swaps
    for k in range(len(neg)-1):
        word = neg.copy()
        word[k], word[k+1] = word[k+1], word[k]
        examples.append(word)
    # adds
    for k in range(len(neg)):
        for l in all_letters:
            word = neg.copy()
            word.insert(k, l)
            examples.append(word)
    # dels
    for k in range(len(neg)):
        word = neg.copy()
        del word[k]
        examples.append(word)
    # replaces
    for k in range(len(neg)):
        for l in all_letters:
            word = neg.copy()
            word[k] = l
            examples.append(word)
    inputs = torch.empty(len(examples), max(
        len(word)+1, min_len), device=args.device, dtype=torch.long)
    inputs[:] = n_chars
    for i, example in enumerate(examples, 0):
        idxs = [letterToIndex(l) for l in example]
        inputs[i][:len(example)] = torch.tensor(idxs, device=args.device, type=torch.long)
    return inputs
