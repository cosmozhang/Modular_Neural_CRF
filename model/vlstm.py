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


class VLSTMCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, dropout=0.):
        super(VLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W = Parameter(torch.Tensor(4, input_size, hidden_size))
        self.U = Parameter(torch.Tensor(4, hidden_size, hidden_size))
        self.b = Parameter(torch.Tensor(4, hidden_size))

        self._h_dropout_mask = None

        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.W)
        init.orthogonal_(self.U)
        self.b.data.zero_()
        self.b.data[0].fill_(1.)

    def set_dropout_masks(self, batch_size):
        # print batch_size
        self.batch_size = batch_size
        if self.dropout:
            if self.training:
                '''
                input dropout mask
                '''
                # self._input_dropout_mask = Variable(torch.bernoulli(torch.Tensor(batch_size, self.input_size).fill_(1 - self.dropout)), requires_grad=False)
                '''
                hidden to hidden dropout mask
                '''
                # print self.dropout
                self._h_dropout_mask = Variable(torch.bernoulli(torch.Tensor(4, batch_size, self.hidden_size).fill_(1 - self.dropout)), requires_grad=False)

            else:
                self._h_dropout_mask = Variable(torch.Tensor(4, batch_size, self.hidden_size).fill_(1. - self.dropout), requires_grad=False)
        else:
            '''
            no dropout in testing mode
            '''
            self._h_dropout_mask = Variable(torch.Tensor(4, batch_size, self.hidden_size).fill_(1.), requires_grad=False)

        '''
        for gpu
        '''
        if torch.cuda.is_available():
            # self._input_dropout_mask = self._input_dropout_mask.cuda()
            self._h_dropout_mask = self._h_dropout_mask.cuda()

    def forward(self, X, hidden_state):
        h_tm1, c_tm1 = hidden_state

        # if self._input_dropout_mask is None:
            # self.set_dropout_masks(X.size(0))

        # def get_mask_slice(mask, idx):
            # if isinstance(mask, list): return mask[idx]
            # else: return mask[idx][-X.size(0):]

        # pdb.set_trace()
        mask = self._h_dropout_mask[:,-X.size(0):]

        h_t = torch.matmul(h_tm1 * mask, self.U)

        # print h_t.size()
        x_t = torch.matmul(X, self.W) + self.b.unsqueeze(1)

        i_t = F.sigmoid(x_t[0] + h_t[0])
        f_t = F.sigmoid(x_t[1] + h_t[1])
        c_t = f_t * c_tm1 + i_t * F.tanh(x_t[2] + h_t[2])
        o_t = F.sigmoid(x_t[3] + h_t[3])
        h_t = o_t * F.tanh(c_t)

        return h_t, c_t


class VLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional=False, dropout=0.0, cell_class=VLSTMCell):
        super(VLSTM, self).__init__()
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

    def forward(self, packed_input, hidden_state=None):
        # print self.training
        is_packed = isinstance(packed_input, PackedSequence)
        if is_packed:
            input_data, batch_sizes = packed_input
            max_batch_size = batch_sizes[0]
        else:
            raise NotImplementedError()

        for cell in self.lstm_cells:
            cell.set_dropout_masks(max_batch_size)

        if hidden_state is None:
            '''
            hidden states
            '''
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input_data.data.new(num_directions,
                                                        max_batch_size,
                                                        self.hidden_size).zero_())

            hidden_state = (hx, hx)

        if self.bidirectional:
            layer = (variable_recurrent_factory(lambda x, h: self.cell(x, h)),
                     variable_recurrent_factory(lambda x, h: self.cell_reverse(x, h), reverse=True))
        else:
            layer = (variable_recurrent_factory(lambda x, h: self.cell(x, h)),)

        func = StackedRNN(layer,
                          num_layers=1,
                          lstm=True,
                          dropout=0.,
                          train=self.training)
        # pdb.set_trace()
        next_hidden, output = func(input_data, hidden_state, [[] for i in range(num_directions)], batch_sizes)

        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, next_hidden


def main():
    """
    Testing
    """

    seqs = ['ghatmasala','nicela','chutpakodas']

    # print list(set(flatten(seqs)))


    # make <pad> idx 0
    vocab = ['<pad>'] + sorted(list(set(list(itertools.chain.from_iterable(seqs))))) # dictionary
    # print vocab


    # make model
    embed = nn.Embedding(len(vocab), 10).cuda()
    lstm = VLSTM(10, 5, bidirectional=True, dropout = 0.5).cuda()

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
    # lstm.train()
    packed_output, (ht, ct) = lstm.forward(packed_input)

    # unpack your output if required
    output, _ = pad_packed_sequence(packed_output)
    # print output

    # pdb.set_trace()
    # Or if you just want the final hidden state?
    print ht[-1]

if __name__ == '__main__':
    main()
