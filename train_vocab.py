import argparse
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from loss import Energy_Loss
from torch.utils.data import DataLoader
from utils import *


def train(args, model, optimizer, criterion, train_loader, epoch):
    model.train()
    for batch_idx, (input, target) in enumerate(train_loader):
        if args.use_cuda:
            input, target = input.cuda(args.device, non_blocking=True), target.cuda(args.device, non_blocking=True)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx) * train_loader.batch_size % args.log_rate == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, (batch_idx) * (train_loader.batch_size), len(train_loader.dataset),
                100. * (batch_idx) / len(train_loader), loss.item()/train_loader.batch_size), end='')
    print()


# train test split.
def test(args, model, criterion, test_loader, n=1000):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if args.use_cuda:
                input, target = input.cuda(args.device, non_blocking=True), target.cuda(args.device, non_blocking=True)
            outputs = model(input)
            test_loss += criterion(outputs, target)
            v, j = outputs.min(1)
            # correct ones are 0.
            correct += test_loader.batch_size - torch.nonzero(j).size()[0]
            total += test_loader.batch_size
    print("Test: Avg Loss: {:.3f}, Accuracy: {:.3f}".format(
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
    parser.add_argument('--topk', type=int, default=50000,
                        help="train on top k most common words in vocabulary (default: 50000)")
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--train_num_neg', type=int, default=31,
                        help='number of negative examples in each training batch (default: 31)')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='number of examples in each batch (default: 25)')
    parser.add_argument('--test_num_neg', type=int, default=9,
                        help='number of negative examples in each test (default: 9)')
    parser.add_argument('--beta', type=float, default=1, metavar='B',
                        help='Inverse Temperature value for Energy function (default: 1)')
    parser.add_argument('--log_rate', type=float, default=1000,
                        help='number of samples per log (default: 1000)')
    parser.add_argument('--num_workers', type=int, default=15,
                        help='number of dataloader workers (default: 15)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.use_cuda else {}
    print("Using Device: {} with {} workers".format(args.device, args.num_workers))

    # instantiate CNN, loss, and optimizer.
    print("Initializing Model...")
    model = CNN(n_chars, 10, 1, 512, [1, 2, 3, 4, 5, 6, 7, 8], 0.25, 1).to(device=args.device)
    print("Initializing Energy Loss with beta = {}".format(args.beta))
    criterion = Energy_Loss(beta=args.beta)
    print("Initializing Adam optimizer with lr = {}".format(args.lr))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    vocab, freq_dict = read_vocab(args.vocab_file, topk=args.topk)
    print("Using wikipedia vocab of size = {}".format(len(vocab)))

    if args.ttsplit:
        print("Train/Test Splitting - Enabled")
        # default use 0.8 train-test split...
        trainset, testset = traintest_split(vocab, p=0.8)
    else:
        print("Train/Test Splitting - Disabled")
        trainset, testset = vocab, vocab

    train_dset = Dataset(trainset, args.train_num_neg)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)

    test_dset = Dataset(testset, args.test_num_neg)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, criterion, train_loader, epoch)
        test(args, model, criterion, test_loader)
        print("Total time: {} sec".format(time.time()-start))
    torch.save(model.state_dict(), args.model_save_file)


if __name__ == '__main__':
    main()
