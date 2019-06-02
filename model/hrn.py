"""
Xiao (Cosmo) Zhang
Thank Pengcheng Yin for providing this code piece
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable
from torch.nn import Parameter, init
from torch.nn._functions.rnn import variable_recurrent_factory, StackedRNN
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import numpy as np
import itertools
import pdb


class HRNNCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, dropout=0.):
        super(HRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W_h = Parameter(torch.Tensor(hidden_size, input_size))
        self.R_h = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = Parameter(torch.Tensor(hidden_size))

        self.W_t = Parameter(torch.Tensor(hidden_size, input_size))
        self.R_t = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_t = Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform(self.W_h)
        init.xavier_uniform(self.R_h)

        init.xavier_uniform(self.W_t)
        init.xavier_uniform(self.R_t)

        self.b_h.data.zero_()
        self.b_t.data.zero_()

    def forward(self, input, hidden_state):
        cs = hidden_state

        # pdb.set_trace()

        # h_t = F.tanh(F.linear(input, self.W_h) + F.linear(cs, self.R_h))
        h_t = F.tanh(F.linear(input, self.W_h) + F.linear(cs, self.R_h) + self.b_h)


        t_t = F.sigmoid(F.linear(input, self.W_t) + F.linear(cs, self.R_t) + self.b_t)

        ns = (h_t-cs)*t_t + cs

        return ns


class HRNN(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional=False, dropout=0.0, cell_class=HRNNCell):
        super(HRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.cell_factory = cell_class
        num_directions = 2 if bidirectional else 1
        self.lstm_cells = []

        for direction in range(num_directions):
            cell = cell_class(input_size, hidden_size, dropout=dropout)
            self.lstm_cells.append(cell)

            suffix = '_reverse' if direction == 1 else ''
            cell_name = 'cell{}'.format(suffix)
            self.add_module(cell_name, cell)

    def forward(self, input, hidden_state=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            raise NotImplementedError()

        '''
        for cell in self.lstm_cells:
            cell.set_dropout_masks(max_batch_size)
        '''

        if hidden_state is None:
            num_directions = 2 if self.bidirectional else 1
            sx = torch.autograd.Variable(input.data.new(num_directions,
                                                        max_batch_size,
                                                        self.hidden_size).zero_())

            hidden_state = sx

        rec_factory = variable_recurrent_factory(batch_sizes)
        if self.bidirectional:
            layer = (rec_factory(lambda x, h: self.cell(x, h)),
                     rec_factory(lambda x, h: self.cell_reverse(x, h), reverse=True))
        else:
            layer = (rec_factory(lambda x, h: self.cell(x, h)),)

        func = StackedRNN(layer,
                          num_layers=1,
                          lstm=False,
                          dropout=0.,
                          train=self.training)

        # pdb.set_trace()
        next_hidden, output = func(input, hidden_state, weight=[[] for i in range(num_directions)])

        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, next_hidden

def flatten(l):
    return list(itertools.chain.from_iterable(l))

def main():
    """
    Testing
    """

    seqs = ['ghatmasala','nicela','chutpakodas']

    # print list(set(flatten(seqs)))


    # make <pad> idx 0
    vocab = ['<pad>'] + sorted(list(set(flatten(seqs)))) # dictionary
    # print vocab


    # make model
    embed = nn.Embedding(len(vocab), 10).cuda()
    lstm = HRNN(10, 5, bidirectional=True).cuda()

    vectorized_seqs = [[vocab.index(tok) for tok in seq]for seq in seqs] # numerialized data

    # print vectorized_seqs



    # get the length of each seq in your batch
    seq_lengths = torch.cuda.LongTensor(map(len, vectorized_seqs))

    # pdb.set_trace()

    # dump padding everywhere, and place seqs on the left.
    # NOTE: you only need a tensor as big as your longest sequence
    seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long().cuda()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)


    # SORT YOUR TENSORS BY LENGTH!
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
    # Otherwise, give (L,B,D) tensors
    seq_tensor = seq_tensor.transpose(0,1) # (B,L,D) -> (L,B,D)

    # embed your sequences
    seq_tensor_emb = embed(seq_tensor)

    # pack them up nicely
    packed_input = pack_padded_sequence(seq_tensor_emb, seq_lengths.cpu().numpy())

    # throw them through your LSTM (remember to give batch_first=True here if you packed with it)
    # pdb.set_trace()
    packed_output, (ht, ct) = lstm.forward(packed_input)

    # unpack your output if required
    output, _ = pad_packed_sequence(packed_output)
    print output

    # pdb.set_trace()
    # Or if you just want the final hidden state?
    print ht[-1]

if __name__ == '__main__':
    main()
