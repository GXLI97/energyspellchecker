import torch
import torch.nn.functional as F


class Energy_Loss(torch.nn.Module):

    def __init__(self, beta=1):
        super(Energy_Loss, self).__init__()
        self.B = beta

    def forward(self, x, y):
        y = y.type(torch.float)
        loss1 = torch.sum(x*y)
        loss2 = 1/self.B * torch.sum(torch.logsumexp(-self.B * x, dim=1))
        loss = loss1 + loss2
        return loss