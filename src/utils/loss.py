
from torch import nn,Tensor
import torch
import numpy as np
from torch.nn.functional import normalize



def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """

    return im.mm(s.t())


def euclidean_sim(x, y):
    """
      Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
      Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    return 1 - dist

class func_CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(func_CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p =  -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class circleLoss(nn.Module):
    """Triplet loss class

    Parameters
    ----------
    margin : float
        Ranking loss margin
    gamma : float
           
    metric : string
        Distance metric (either euclidean or cosine)
    """

    def __init__(self, margin=0.25, gamma=256,metric='cosine'):

        super(circleLoss, self).__init__()
        self.distance_function = euclidean_sim if metric == 'euclidean' else cosine_sim
        self.metric = metric
        self.func_circle_loss = func_CircleLoss(m = margin,gamma = gamma)
        #self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(self, im, s):
        # compute image-sentence score matrix
        # batch_size x batch_size
        scores_i2r = self.distance_function(normalize(im, dim=-1),
                                            normalize(s, dim=-1))
        scores_r2i = scores_i2r.t()
        pos = torch.eye(im.size(0))
        neg = 1 - pos
         
        #pos = (pos == 1).to(im.device)
        #neg = (neg == 1).to(im.device)
        pos = pos.bool().to(im.device)
        neg = neg.triu(diagonal=1).bool().to(im.device)
      
        scores_i2r = scores_i2r.reshape(-1)
        scores_r2i = scores_r2i.reshape(-1)
        # positive similarities
        # batch_size x 1
        sp1 = scores_i2r[pos.reshape(-1)]
        sp2 = scores_r2i[pos.reshape(-1)]
        
        #negative _matrix
        sn1 = scores_i2r[neg.reshape(-1)]
        sn2 = scores_r2i[neg.reshape(-1)]
        
        cost_im = self.func_circle_loss(sp1,sn1)
        cost_s = self.func_circle_loss(sp2,sn2)
        # clear diagonals
        
       
        return (cost_s + cost_im)
    

if __name__ == "__main__":

    feat1 = nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
    feat2 = nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
    
    criterion = circleLoss(margin=0.25, gamma=256) 
    circle_loss = criterion(feat1, feat2)

    
