import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from loss import Energy_Loss
from utils import *


def train(args, model, optimizer, criterion, data, epoch):
    model.train()
    running_loss = 0.0
    for i, word in enumerate(data, 0):
        if len(word) < 3:
            continue
        inputs, labels = minibatch(args, word, data, num_neg=args.num_neg)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:
            print('\r[%d, %5d] Avg Loss: %.3f' %
                  (epoch, i+1, running_loss/(i+1)), end='')
    print()


# train test split.
def test(args, model, criterion, data, n=1000):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for i in range(n):
            word = data[random.randint(0, len(data)-1)]
            if len(word) < 3:
                continue
            inputs, labels = minibatch(args, word, data, num_neg=9)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels)

            v, i = outputs.min(0)
            # first element is smallest
            if i == 0:
                correct += 1
            total += 1
    print("Avg Loss: {:.3f}, Accuracy: {:.3f}".format(
        test_loss/total, correct/total))


def main():
    parser = argparse.ArgumentParser(
        description='Training Energy NN for Spell Correction')
    parser.add_argument('--vocab_file', type=str, default="data/wikipedia.vocab.txt",
                        help='vocabulary file path')
    parser.add_argument('--model_save_file', type=str, default='models/energy_cnn',
                        help='model save path')
    parser.add_argument('--ttsplit', action='store_true', default=False,
                        help='enables train/test splitting (80-20)')
    parser.add_argument('--topk', type=int, default=10000,
                        help="train on top k most common words in vocabulary")
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--num_neg', type=int, default=15,
                        help='number of negative examples in each batch (default: 15)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    # instantiate CNN, loss, and optimizer.
    model = CNN(n_chars, 10, 1, 256, [1, 2, 3, 4, 5, 6], 0.1, 1).to(device=args.device)
    criterion = Energy_Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    vocab, freq_dict = read_vocab(args.vocab_file, topk=args.topk)

    if args.ttsplit:
        # default use 0.8 train-test split...
        trainset, testset = traintest_split(vocab, p=0.8)
    else:
        trainset, testset = vocab, vocab

    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, criterion, trainset, epoch)
        test(args, model, criterion, testset)

    torch.save(model.state_dict(), args.model_save_file)


if __name__ == '__main__':
    main()
