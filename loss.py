import torch
import torch.nn.functional as F


class Energy_Loss(torch.nn.Module):

    def __init__(self, Beta=5):
        super(Energy_Loss, self).__init__()
        self.B = Beta

    def forward(self, x, y):
        # calculate some loss.
        x = x.squeeze(1)
        # normalize trick to ensure that x predictions don't blow up.
        # x = x/torch.norm(x, p=2)
        free_energy = 1/self.B * torch.logsumexp(-self.B*x, dim=0)
        return x[0] + free_energy
        # x = F.log_softmax(x, dim=0)
        # print(x)
        # y = y.type(torch.float)
        # totloss = torch.sum(x*y)
        # totloss = x[0]
        # print(totloss)
        # return totloss
