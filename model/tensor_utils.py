import torch.nn as nn
import torch.nn.init
import numpy as np
import pdb

def to_scalar(var):
    """change the first element of a tensor to scalar
    """
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    """helper function to calculate argmax of input vector at dimension 1
    """
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def log_sum_exp(tensor, dim  = None):
    # Compute log sum exp in a numerically stable way for the forward algorithm

    xmax, _ = torch.max(tensor, dim = dim, keepdim = True)
    xmax_, _ = torch.max(tensor, dim = dim)
    return xmax_ + torch.log(torch.sum(torch.exp(tensor - xmax), dim = dim))

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    # bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    # nn.init.xavier_uniform_(input_linear.weight.data)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    # weight = eval('input_lstm.weight_ih_l0')
    # bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
    # pdb.set_trace()
    nn.init.orthogonal_(input_lstm.weight_ih_l0.data)

    # weight = eval('input_lstm.weight_hh_l0')
    # bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
    nn.init.orthogonal_(input_lstm.weight_hh_l0.data)

    if input_lstm.bidirectional:
        # weight = eval('input_lstm.weight_ih_l0_reverse')
        # bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.orthogonal_(input_lstm.weight_ih_l0_reverse.data)

        # weight = eval('input_lstm.weight_hh_l0_reverse')
        # bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.orthogonal_(input_lstm.weight_hh_l0_reverse.data)

    if input_lstm.bias:
        # weight = eval('input_lstm.bias_ih_l'+str(0))
        input_lstm.bias_ih_l0.data.zero_()
        # weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        # weight = eval('input_lstm.bias_hh_l'+str(0))
        input_lstm.bias_hh_l0.data.zero_()
        # weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        if input_lstm.bidirectional:

            input_lstm.bias_ih_l0_reverse.data.zero_()

            input_lstm.bias_hh_l0_reverse.data.zero_()
