import string
import random
import math
import torch

all_letters = string.ascii_letters + " .,;'"
n_chars = len(all_letters)

def read_vocab(filename,topk=10000):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    lines = [line.split() for line in lines][:topk]
    random.shuffle(lines)
    vocab = [line[0] for line in lines]
    freq_dict = {line[0]: int(line[1]) for line in lines}
    return vocab, freq_dict

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

def traintest_split(data, p=0.8):
    idx = math.floor(len(data)*p)
    trainset = data[:idx]
    testset = data[idx:]
    return trainset, testset


def minibatch(line, num_neg=2, min_len=7):
    # one positive and num_neg negative examples.
    idxs = [letterToIndex(l) for l in line]
    inputs = torch.tensor(idxs, dtype=torch.long).unsqueeze(0)
    for i in range(num_neg):
        # swap adjacent characters randomly.
        k = random.randint(0, len(idxs)-2)
        if idxs[k] != idxs[k+1]:
            swapped_idxs = idxs.copy()
            swapped_idxs[k], swapped_idxs[k +
                                          1] = swapped_idxs[k+1], swapped_idxs[k]
            temp = torch.tensor(swapped_idxs, dtype=torch.long).unsqueeze(0)
            inputs = torch.cat((inputs, temp), 0)
    if inputs.size(1) < min_len:
        pad = n_chars * torch.ones(inputs.size(0),
                                   min_len-inputs.size(1), dtype=torch.long)
        inputs = torch.cat((inputs, pad), 1)
    labels = torch.zeros(inputs.size(0), dtype=torch.long)
    # 1 for correct, -1 for incorrect.
    labels[0] = 1
    labels[1:] = -1
    return inputs, labels


def test_accuracy(data, cnn, n):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(n):
            line = data[random.randint(0, len(data)-1)]
            if len(line) < 3:
                continue
            inputs, labels = minibatch(line, num_neg=9)
            outputs = cnn(inputs)
            v, i = outputs.min(0)
            # first element is smallest
            if i == 0:
                correct += 1
            total += 1
    print("Accuracy: {}".format(correct/total))
