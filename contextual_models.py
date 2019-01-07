import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim


class CBOW(nn.Module):
    def __init__(self, weights_matrix, context_size):
        super(CBOW, self).__init__()
        self.glove_embed, self.vocab_size, self.embed_dim = create_emb_layer(weights_matrix)
        self.fc1 = nn.Linear(context_size * self.embed_dim, 100)
        self.fc2 = nn.Linear(100, self.vocab_size)
        
    def forward(self, inputs):
        x = self.glove_embed(inputs).view((1,-1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        log_probs = F.log_softmax(x, dim=1)
        return log_probs

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
        self.fc1 = nn.Linear(len(Ks)*Co, 1000)
        self.fc2 = nn.Linear(1000, Out)

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
        # TODO: try to normalize?
        # x = (x-torch.mean(x))/torch.norm(x) # normalize
        # unsquash.
        x = x.view(input.size(0),input.size(1)) # (B, N, Out)
        return x


class contextCNN(nn.Module):
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
        self.fc1 = nn.Linear(len(Ks)*Co, 1000)
        self.fc2 = nn.Linear(1000, Out)

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
        # TODO: try to normalize?
        # x = (x-torch.mean(x))/torch.norm(x) # normalize
        # unsquash.
        x = x.view(input.size(0),input.size(1)) # (B, N, Out)
        return x