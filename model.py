import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        self.embedding = nn.Embedding(Nc, D, padding_idx=0)
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(Ci, Co, kernel_size=(K, D)) for K in Ks])
        self.dropout = nn.Dropout(Drop)
        self.fc1 = nn.Linear(len(Ks)*Co, 500)
        self.fc2 = nn.Linear(500, Out)

    def forward(self, input):
        # input is (B, N, W)
        # N is number of examples in minibatch
        # W is length of each example (maximum word length)
        # squash the batche examples into one dimension (B*N, W)
        x = input.view(-1,input.size(2))
        x = self.embedding(x)  # (B*N, W, D)
        x = x.unsqueeze(1)  # (B*N, 1, W, D)
        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs1]  # [(B*N, Co, Lk) for K in Ks]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(B*N, Co) for K in Ks]
        x = torch.cat(x, 1)  # (B*N, Co * len(Ks))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))  # (B*N, 500)
        x = self.fc2(x) #(B*N, Out)
        # unsquash.
        x = x.view(input.size(0),input.size(1)) # (B, N, Out)
        return x
