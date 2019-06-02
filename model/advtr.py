import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import pdb

def _l2_normalize(grad, mask):

    newmask = mask.unsqueeze(-1)
    # pdb.set_trace()
    g = grad/torch.sqrt(torch.sum(torch.masked_select(grad, newmask.expand_as(grad)) ** 2) + 1e-16)
    g = g.clone()
    g.requires_grad = False

    '''
    0 axis is kept
    '''

    '''
    return a tensor
    '''
    return g

def cal_adv(X, mask, epsilon=1.0):
    # pdb.set_trace()
    eadv = epsilon * _l2_normalize(X.grad, mask)
    return eadv


