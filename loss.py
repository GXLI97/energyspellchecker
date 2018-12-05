import torch
import torch.nn.functional as F


class Energy_Loss(torch.nn.Module):

    def __init__(self, Beta=1):
        super(Energy_Loss, self).__init__()
        self.B = Beta

    def forward(self, x, y):
        # calculate some loss.
        x = x.squeeze(1)
        y = y.type(torch.float)
        # normalize trick to ensure that x predictions don't blow up.
        # x = x/torch.norm(x, p=2)
        loss1 = torch.sum(x*y)/torch.sum(y)
        loss2 = 1/self.B + torch.logsumexp(-self.B * x, dim=0)
        # loss2 = -torch.logsumexp(x,dim=0)
        loss = loss1 + loss2
        # print("{:.2f} {:.2f}".format(loss1.item(), loss2.item()))
        return loss