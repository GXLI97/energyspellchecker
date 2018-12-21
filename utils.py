import string
import random
import math
import itertools
import torch
from torch.utils import data


all_letters = ['<pad>'] + list(string.ascii_lowercase)
tok2index = {k: v for v, k in enumerate(all_letters)}
n_chars = len(all_letters)
min_len = 8

def read_vocab(filename, topk=10000):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    lines = [line.split() for line in lines]
    vocab = [line[0] for line in lines if len(line[0]) >= 3][:topk]
    freq_dict = {line[0]: int(line[1]) for line in lines}
    return vocab, freq_dict


def write_vocab_to_file(dataset, freq_dict, save_file):
    with open(save_file, 'w') as f:
        for word in dataset:
            f.write(word + " " + str(freq_dict[word]) + "\n")


def traintest_split(data, p=0.8):
    random.shuffle(data)
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


def get_random_negative(word, vocab, edit):
    neg = word.copy()
    if edit == 0:
        edits = [r_swap, r_add, r_del, r_replace]
    elif edit == 1:
        edits = [r_swap]
    elif edit == 2:
        edits = [r_add, r_del]
    else:
        edits = [r_replace]
    neg = random.choice(edits)(neg)
    if "".join(neg)  in vocab:
        # just return null example.
        neg = []
    # print("".join(neg))
    return neg


def build_all(neg, edit):
    examples = []
    neg = list(neg)
    examples.append(neg)
    # swaps->swaps
    if edit == 0 or edit == 1:
        for k in range(len(neg)-1):
            word = neg.copy()
            word[k], word[k+1] = word[k+1], word[k]
            examples.append(word)
    # add/del
    if edit == 0 or edit == 2:
        for k in range(len(neg)):
            for l in all_letters:
                word = neg.copy()
                word.insert(k, l)
                examples.append(word)
        for k in range(len(neg)):
            word = neg.copy()
            del word[k]
            examples.append(word)
    # replaces->replaces
    if edit == 0 or edit == 3:
        for k in range(len(neg)):
            for l in all_letters:
                word = neg.copy()
                word[k] = l
                examples.append(word)
    inputs = torch.zeros(1, len(examples), max(
        len(neg)+1, min_len), dtype=torch.long)
    for i, ex in enumerate(examples, 0):
        vec_examples = [tok2index[tok] for tok in ex]
        inputs[0][i][:len(ex)] = torch.tensor(
            vec_examples, dtype=torch.long)
    return inputs


class Dataset(data.Dataset):

    def __init__(self, vocab, num_neg, edit):
        super(Dataset, self).__init__()
        self.vocab = vocab
        self.vocabset = set(vocab)
        self.num_neg = num_neg
        self.edit = edit

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, index):
        word = self.vocab[index]
        inputs, labels = nce(word, self.vocabset, self.num_neg, self.edit)
        return inputs, labels


def nce(word, vocab, num_neg, edit):
    examples = []
    word = list(word)
    examples.append(word)
    while len(examples) != num_neg + 1:
        neg = get_random_negative(word, vocab, edit)
        examples.append(neg)
    vec_examples = [[tok2index[tok] for tok in ex] for ex in examples]
    # print(vec_examples)
    labels = torch.zeros(len(examples), dtype=torch.long)
    labels[0] = 1
    # print(labels)
    return vec_examples, labels


def collate_fn(batch):
    # B X E X MaxD.
    d1 = len(batch)  # args.batch_size
    d2 = len(batch[0][0])  # 1 + args.num_neg
    # max length of words in batch.
    d3 = max([len(ex) for exs in batch for ex in exs[0]])

    inputs = torch.zeros(d1, d2, max(d3, min_len), dtype=torch.long)
    # UNDER CONSTRUCTION
    for idx1, b in enumerate(batch):
        vec_lengths = torch.tensor([len(seq)
                                    for seq in b[0]], dtype=torch.long)
        for idx2, (vec, veclen) in enumerate(zip(b[0], vec_lengths)):
            inputs[idx1][idx2][:veclen] = torch.tensor(vec, dtype=torch.long)
    # B X E
    targets = torch.cat([b[1].unsqueeze(0) for b in batch], 0)
    # print(inputs)
    # print(targets)
    return inputs, targets
