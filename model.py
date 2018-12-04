import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_chars, embed_dim, channel_in, channel_out, kernel_sizes, dropout, output_dim):
        super(CNN, self).__init__()
        Nc = n_chars
        D = embed_dim
        Ci = channel_in
        Co = channel_out
        Ks = kernel_sizes
        Drop = dropout
        Out = output_dim
        # construct embedding lookup table.
        # use last one as data pad.
        self.embedding = nn.Embedding(Nc+1, D, padding_idx=Nc)
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(Ci, Co, kernel_size=(K, D)) for K in Ks])
        self.dropout = nn.Dropout(Drop)
        self.fc1 = nn.Linear(len(Ks)*Co, Out)

    def forward(self, input):
        # input is (N, W)
        # N is number of examples in minibatch
        # W is length of each example (maximum word length)
        x = self.embedding(input.squeeze(0))  # (N, W, D)
        x = x.unsqueeze(1)  # (N, 1, W, D)
        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs1]  # [(N, Co, Lk) for K in Ks]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co) for K in Ks]
        x = torch.cat(x, 1)  # (N, Co * len(Ks))
        x = self.dropout(x)
        x = self.fc1(x)  # (N, Out)
        return x
