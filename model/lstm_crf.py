"""

"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import model.tensor_utils as tensor_utils
import model.highway as highway
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
import pdb

class LSTM_CRF(nn.Module):
    """
    two-headed lstm-crf
    """

    def __init__(self, tagset_size, char_size, char_dim, char_hidden_dim, embedding_dim, word_hidden_dim, win_size, vocab_size, dropout_ratio, tag_dim = 100, segtgt_size = None, enttgt_size = None, if_highway = False, ex_embedding_dim = None, segment_loss=0, entity_loss=0):

        super(LSTM_CRF, self).__init__()

        self.xentropy = nn.CrossEntropyLoss(size_average=False)

        self.char_dim = char_dim
        self.char_hidden_dim = char_hidden_dim
        self.char_size = char_size
        self.word_dim = embedding_dim
        self.ex_word_dim = ex_embedding_dim
        self.win_size = win_size
        self.word_hidden_dim = word_hidden_dim
        self.tag_dim = tag_dim
        self.word_size = vocab_size
        self.if_highway = if_highway
        self.char_embeds = nn.Embedding(char_size, char_dim)
        self.segment_loss = segment_loss
        self.entity_loss = entity_loss
        self.W1 =  nn.Parameter(torch.zeros(word_hidden_dim, word_hidden_dim))
        self.W2 =  nn.Parameter(torch.zeros(word_hidden_dim, word_hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(word_hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(word_hidden_dim))

        self.seg_word_hidden_dim = word_hidden_dim

        self.ent_word_hidden_dim = word_hidden_dim

        self.forw_char_lstm = nn.LSTM(char_dim, char_hidden_dim, bidirectional=False, dropout=dropout_ratio)
        self.back_char_lstm = nn.LSTM(char_dim, char_hidden_dim, bidirectional=False, dropout=dropout_ratio)
        if not ex_embedding_dim:
            self.word_lstm = nn.LSTM(self.word_dim * self.win_size + char_hidden_dim * 2, self.word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)
        else:
            '''
            use two embeddings
            '''
            self.word_lstm = nn.LSTM((self.word_dim + self.ex_word_dim) * self.win_size + char_hidden_dim * 2, self.word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        if self.ex_word_dim > 0:
            self.ex_word_embeds = nn.Embedding(vocab_size, self.ex_word_dim)
        else:
            self.ex_word_embeds = None

        self.dropout = nn.Dropout(p=dropout_ratio)

        '''
        highway nets
        '''
        if if_highway:
            self.fbchar_highway = highway.hw(2 * char_hidden_dim, dropout_ratio=dropout_ratio)

        self.tag_size = tagset_size
        self.seg_size = segtgt_size
        self.ent_size = enttgt_size

        self.small = -1000

        if self.segment_loss != 2 and self.entity_loss != 2:
            self.hidden2tag = nn.Linear(self.word_hidden_dim, self.tag_size)
        elif self.segment_loss == 2 and self.entity_loss !=2:
            self.hidden2tag = nn.Linear(self.word_hidden_dim+self.seg_word_hidden_dim, self.tag_size)
        elif self.segment_loss != 2 and self.entity_loss ==2:
            self.hidden2tag = nn.Linear(self.word_hidden_dim+self.ent_word_hidden_dim, self.tag_size)
        elif self.segment_loss == 2 and self.entity_loss ==2:
            self.hidden2tag = nn.Linear(self.word_hidden_dim+self.ent_word_hidden_dim+self.seg_word_hidden_dim, self.tag_size)
            '''
            bilinear layer
            '''
            # self.bilinear = nn.Bilinear(self.word_hidden_dim, self.word_hidden_dim, self.tag_size)


        if self.segment_loss != 0 :
            self.segtgt_size = segtgt_size
            if not ex_embedding_dim:
                self.word_lstm_seg = nn.LSTM(self.word_dim * self.win_size + char_hidden_dim * 2, self.seg_word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)
            else:
                '''
                use two embeddings
                '''
                self.word_lstm_seg = nn.LSTM((self.word_dim + self.ex_word_dim) * self.win_size + char_hidden_dim * 2, self.seg_word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)
            self.hidden2seg = nn.Linear(self.seg_word_hidden_dim, self.segtgt_size)
            self.seg_transitions = nn.Parameter(torch.zeros(self.segtgt_size+2, self.segtgt_size+2))

        if self.entity_loss != 0:
            self.enttgt_size = enttgt_size
            if not ex_embedding_dim:
                self.word_lstm_ent = nn.LSTM(self.word_dim * self.win_size + char_hidden_dim * 2, self.ent_word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)
            else:
                '''
                use two embeddings
                '''
                self.word_lstm_ent = nn.LSTM((self.word_dim + self.ex_word_dim) * self.win_size + char_hidden_dim * 2, self.ent_word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)
            self.hidden2ent = nn.Linear(self.ent_word_hidden_dim, self.enttgt_size)
            self.ent_transitions = nn.Parameter(torch.zeros(self.enttgt_size+2, self.enttgt_size+2))

        self.transitions = nn.Parameter(torch.zeros(self.tag_size+2, self.tag_size+2))

        self.rand_init()

    def load_word_embedding(self, pre_word_embeddings, no_fine_tune, extra = False):

        # assert (pre_word_embeddings.size()[1] == self.word_dim)
        if no_fine_tune:
            if not extra:
                self.word_embeds.weight = autograd.Variable(pre_word_embeddings)
            else:
                self.ex_word_embeds.weight = autograd.Variable(pre_word_embeddings)
        else:
            if not extra:
                self.word_embeds.weight = nn.Parameter(pre_word_embeddings)
            else:
                self.ex_word_embeds.weight = nn.Parameter(pre_word_embeddings)

    def rand_init(self, init_char_embedding=True):
        """
        """

        if init_char_embedding:
            tensor_utils.init_embedding(self.char_embeds.weight)
        if self.if_highway:
            self.fbchar_highway.rand_init()

        tensor_utils.init_lstm(self.forw_char_lstm)
        tensor_utils.init_lstm(self.back_char_lstm)
        tensor_utils.init_lstm(self.word_lstm)
        tensor_utils.init_linear(self.hidden2tag)

        if self.segment_loss != 0 :
            tensor_utils.init_linear(self.hidden2seg)
            tensor_utils.init_lstm(self.word_lstm_seg)
        if self.entity_loss != 0 :
            tensor_utils.init_linear(self.hidden2ent)
            tensor_utils.init_lstm(self.word_lstm_ent)
        self.transitions.data.zero_()



    def _cal_emission(self, c_f_v, cf_p_v, c_b_v, cb_p_v, word_emb, sent_lens, char_lens):
        '''
        '''

        word_seq_length = cf_p_v.size(0)
        batch_size = cf_p_v.size(1)

        '''
        char embedding layer
        '''
        forw_emb = self.char_embeds.forward(c_f_v)
        back_emb = self.char_embeds.forward(c_b_v)

        '''
        dropout
        '''
        d_f_emb_packed_in = pack_padded_sequence(forw_emb, char_lens)

        d_b_emb_packed_in = pack_padded_sequence(back_emb, char_lens)

        '''
        feed the whole sequence to lstm
        '''
        packed_cf_out, _ = self.forw_char_lstm.forward(d_f_emb_packed_in)  #seq_len_char * batch * char_hidden_dim
        forw_lstm_out, _ = pad_packed_sequence(packed_cf_out)

        packed_cb_out, _ = self.back_char_lstm.forward(d_b_emb_packed_in)  #seq_len_char * batch * char_hidden_dim
        back_lstm_out, _ = pad_packed_sequence(packed_cb_out)

        '''
        select positions
        '''
        cf_p_v = cf_p_v.unsqueeze(2).expand(word_seq_length, batch_size, self.char_hidden_dim)
        select_forw_lstm_out = torch.gather(forw_lstm_out, 0, cf_p_v)

        cb_p_v = cb_p_v.unsqueeze(2).expand(word_seq_length, batch_size, self.char_hidden_dim)
        select_back_lstm_out = torch.gather(back_lstm_out, 0, cb_p_v)

        if self.if_highway:
            fb_lstm_out = self.dropout(torch.cat((select_forw_lstm_out, select_back_lstm_out), dim=2))
            char_out = self.fbchar_highway.forward(fb_lstm_out)
        else:
            char_out = torch.cat((select_forw_lstm_out, select_back_lstm_out), dim=2)

        # char_out = self.dropout.forward(char_out)

        '''
        combine char and word embeddings
        '''
        word_input = torch.cat((word_emb, char_out), dim = 2)

        packed_word_input = pack_padded_sequence(word_input, sent_lens)

        #word level lstm
        packed_lstm_out, _ = self.word_lstm.forward(packed_word_input)

        lstm_out, _ = pad_packed_sequence(packed_lstm_out)

        if self.segment_loss != 0:

            packed_lstm_out_seg, _ = self.word_lstm_seg.forward(packed_word_input)
            lstm_out_seg, _ = pad_packed_sequence(packed_lstm_out_seg)
            # lstm_out_seg = self.dropout.forward(lstm_out_seg)
            seg_scores_out = self.hidden2seg.forward(lstm_out_seg)
        else:
            seg_scores_out = None

        if self.entity_loss != 0:

            packed_lstm_out_ent, _ = self.word_lstm_ent.forward(packed_word_input)
            lstm_out_ent, _ = pad_packed_sequence(packed_lstm_out_ent)
            # lstm_out_ent = self.dropout.forward(lstm_out_ent)
            ent_scores_out = self.hidden2ent.forward(lstm_out_ent)
        else:
            ent_scores_out = None

        if self.segment_loss == 2 and self.entity_loss == 2:
            lstm_out_seg = torch.sigmoid(torch.matmul(lstm_out, self.W1) + self.b1) * lstm_out_seg
            lstm_out_ent = torch.sigmoid(torch.matmul(lstm_out, self.W2) + self.b2) * lstm_out_ent
            lstm_out = torch.cat((lstm_out_seg, lstm_out, lstm_out_ent), dim = 2)
            # scores_out = self.bilinear.forward(lstm_out_seg, lstm_out_ent)

        scores_out = self.hidden2tag.forward(lstm_out) # seq_len * batch_size * tag_size

        return scores_out, F.relu(seg_scores_out) if self.segment_loss != 0 else None, F.relu(ent_scores_out) if self.entity_loss != 0 else None

    def _forward_alg(self, unaries, transitions, padded_mask):

        '''
        unaries: seq_len+2 * batch_size * tag_size+2
        padded_mask: seq_len+1 * batch_size
        '''

        seq_len = unaries.size(0)
        batch_size = unaries.size(1)
        padded_tag_size = unaries.size(2) # real_tag_size+2
        start_tag = padded_tag_size-2
        end_tag = padded_tag_size - 1

        alpha = unaries[0, :, start_tag].expand(padded_tag_size, -1).clone() # tag_size+2 * batch_size

        transitions = transitions.expand(batch_size, -1, -1).permute(1, 2, 0) # from_tag_size+2 * to_tag_size+2 * batch_size

        for step_idx, (mask_step, unary) in enumerate(zip(padded_mask, unaries[1:])):
            # alpha: tag_size+2 * batch_size
            # unary: batch_size * tag_size+2
            # mask_step: batch_size

            temp = alpha.expand(padded_tag_size, -1, -1).permute(1, 0, 2) + transitions + unary.expand(padded_tag_size, -1, -1).permute(0, 2, 1) # from_tag_size+2 * to_tag_size+2 * batch_size

            # padded_mask[idx]: batch_size
            alpha.masked_scatter_(mask_step.expand(padded_tag_size, -1), tensor_utils.log_sum_exp(temp, dim = 0).masked_select(mask_step.expand(padded_tag_size, -1)))

        final_alpha = alpha[end_tag, :].sum()

        return final_alpha


    def _cal_crf_cost(self, transitions, emission_score, tag_size, mask_v, tg_v, seq_len, batch_size):

        start_tag = tag_size
        end_tag = tag_size + 1

        transitions.data[:, start_tag] = self.small
        transitions.data[end_tag, :] = self.small
        transitions.data[start_tag, end_tag] = self.small

        if torch.cuda.is_available():
            neg_mask = autograd.Variable((1-mask_v.data).cuda())
        else:
            neg_mask = autograd.Variable((1-mask_v.data))
        emission_score.masked_fill_(neg_mask.expand(tag_size, -1, -1).permute(1,2,0), self.small)

        b_s = np.array([[self.small] * tag_size + [0.0, self.small]]).astype(np.float32)
        e_s = np.array([[self.small] * tag_size + [self.small, 0.0]]).astype(np.float32)
        if torch.cuda.is_available():
            b_s_v = autograd.Variable(torch.from_numpy(b_s).expand(1, batch_size, -1).cuda()) # batch_size * tag_size+2
            e_s_v = autograd.Variable(torch.from_numpy(e_s).expand(1, batch_size, -1).cuda()) # batch_size * tag_size+2
        else:
            b_s_v = autograd.Variable(torch.from_numpy(b_s).expand(1, batch_size, -1)) # batch_size * tag_size+2
            e_s_v = autograd.Variable(torch.from_numpy(e_s).expand(1, batch_size, -1)) # batch_size * tag_size+2

        s_padding = self.small*torch.ones(seq_len, batch_size, 1)
        e_padding = torch.zeros(seq_len, batch_size, 1)

        # seq_len * batch_size * 2
        if torch.cuda.is_available():
            s_padding_v = autograd.Variable(s_padding.cuda())
            e_padding_v = autograd.Variable(e_padding.cuda())
        else:
            s_padding_v = autograd.Variable(s_padding)
            e_padding_v = autograd.Variable(e_padding)

        e_padding_v.masked_fill_(mask_v.expand(1, -1, -1).permute(1, 2, 0), self.small)

        padded_emission_score = torch.cat([emission_score, s_padding_v, e_padding_v], dim = 2) # seq_len * batch_size * tag_size+2

        unaries = torch.cat([b_s_v, padded_emission_score, e_s_v], dim = 0) # seq_len+2 * batch_size * tag_size+2

        emi_real_path_score = torch.sum(torch.gather(padded_emission_score, 2, tg_v).masked_select(mask_v.unsqueeze(2)))

        if torch.cuda.is_available():
            b_id_v = autograd.Variable(torch.LongTensor(1, batch_size).fill_(tag_size).cuda()) # 1 * batch_size
            e_id_v = autograd.Variable(torch.LongTensor(1, batch_size).fill_(tag_size+1).cuda()) # 1 * batch_size
        else:
            b_id_v = autograd.Variable(torch.LongTensor(1, batch_size).fill_(tag_size)) # 1 * batch_size
            e_id_v = autograd.Variable(torch.LongTensor(1, batch_size).fill_(tag_size+1)) # 1 * batch_size

        padded_target = torch.cat([b_id_v, tg_v.squeeze(2), e_id_v], dim = 0)
        if torch.cuda.is_available():
            to_pad = autograd.Variable(torch.ByteTensor(1, batch_size).fill_(1).cuda())
        else:
            to_pad = autograd.Variable(torch.ByteTensor(1, batch_size).fill_(1))
        padded_mask = torch.cat([to_pad, mask_v], dim = 0)

        tran_real_path_score = torch.sum(transitions[padded_target[:-1, :], padded_target[1:, :]].masked_select(padded_mask))

        real_score =  emi_real_path_score + tran_real_path_score


        partition_score = self._forward_alg(unaries, transitions, padded_mask)

        cost = - (real_score - partition_score)

        return cost


    def cal_loss(self, c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, seg_tg_v, ent_tg_v, mask_v, sent_lens, char_lens, average_batch = True):

        '''
        tg_v: seq_len * batch_size
        mask_v: seq_len * batch_size
        '''
        seq_len = tg_v.size(0)
        batch_size = tg_v.size(1)

        tag_size = self.tag_size
        seg_size = self.seg_size
        ent_size = self.ent_size


        '''
        word embeddings
        '''
        word_emb = self.dropout.forward(torch.cat([self.word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2))
        if self.ex_word_embeds:
            word_emb2 = self.dropout.forward(torch.cat([self.ex_word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2))
            word_emb = torch.cat([word_emb, word_emb2], dim = 2)
        word_emb.retain_grad()

        emission_score, emission_seg_score, emission_ent_score = self._cal_emission(c_f_v, cf_p_v, c_b_v, cb_p_v, word_emb, sent_lens, char_lens) # seq_len * batch_size * tag_size

        des_cost = self._cal_crf_cost(self.transitions, emission_score, tag_size, mask_v, tg_v, seq_len, batch_size)

        if self.segment_loss != 0:

            seg_loss = self._cal_crf_cost(self.seg_transitions, emission_seg_score, seg_size, mask_v, seg_tg_v, seq_len, batch_size)
        else:
            seg_loss = 0

        if self.entity_loss != 0:

            ent_loss = self._cal_crf_cost(self.ent_transitions, emission_ent_score, ent_size, mask_v, ent_tg_v, seq_len, batch_size)
        else:
            ent_loss = 0

        if average_batch:
            des_cost = des_cost/ batch_size # average over batch
            seg_loss = seg_loss/ batch_size
            ent_loss = ent_loss/ batch_size

        return des_cost, seg_loss, ent_loss, word_emb


    def cal_adv_loss(self, c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, seg_tg_v, ent_tg_v, mask_v, sent_lens, char_lens, eadv, average_batch = True):

        '''
        tg_v: seq_len * batch_size
        mask_v: seq_len * batch_size
        '''
        batch_size = tg_v.size(1)
        tag_size = self.tag_size
        seg_size = self.seg_size
        ent_size = self.ent_size
        seq_len = tg_v.size(0)


        #word
        word_emb = self.dropout.forward(torch.cat([self.word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2))
        if self.ex_word_embeds:
            word_emb2 = self.dropout.forward(torch.cat([self.ex_word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2))
            word_emb = torch.cat([word_emb, word_emb2], dim = 2)
        word_emb += eadv
        emission_score, emission_seg_score, emission_ent_score = self._cal_emission(c_f_v, cf_p_v, c_b_v, cb_p_v, word_emb, sent_lens, char_lens) # seq_len * batch_size * tag_size

        des_cost = self._cal_crf_cost(self.transitions, emission_score, tag_size, mask_v, tg_v, seq_len, batch_size)


        if self.segment_loss != 0:

            seg_loss = self._cal_crf_cost(self.seg_transitions, emission_seg_score, seg_size, mask_v, seg_tg_v, seq_len, batch_size)
        else:
            seg_loss = 0

        if self.entity_loss != 0:

            ent_loss = self._cal_crf_cost(self.ent_transitions, emission_ent_score, ent_size, mask_v, ent_tg_v, seq_len, batch_size)
        else:
            ent_loss = 0

        if average_batch:
            des_cost = des_cost/ batch_size # average over batch
            seg_loss = seg_loss/ batch_size
            ent_loss = ent_loss/ batch_size

        return des_cost, seg_loss, ent_loss

    def _viterbi_decode(self, unaries, transitions, padded_mask):

        '''
        unaries: seq_len * batch_size * tag_size
        '''
        seq_len = unaries.size(0)
        batch_size = unaries.size(1)
        padded_tag_size = unaries.size(2) # real_tag_size+2
        start_tag = padded_tag_size - 2
        end_tag = padded_tag_size - 1

        newmask = 1 - padded_mask


        forward_scores = unaries[0, :, start_tag].expand(padded_tag_size, -1).clone() # tag_size+2 * batch_size

        transitions = transitions.expand(batch_size, -1, -1).permute(1, 2, 0) # from_tag_size * to_tag_size * batch_size

        back_points = [] # seq_len-1 * tag_size+2 * batch_size
        decode_idx = torch.LongTensor(seq_len-1, batch_size).fill_(end_tag) # seq_len-1 * batch_size

        for step_idx, (mask_step, unary) in enumerate(zip(newmask, unaries[1:])):
            # forward_scores: tag_size+2 * batch_size
            # unary: batch_size * tag_size+2
            # mask_step: batch_size

            temp = forward_scores.expand(padded_tag_size, -1, -1).permute(1, 0, 2) + transitions + unary.expand(padded_tag_size, -1, -1).permute(0, 2, 1) # from_tag_size+2 * to_tag_size+2 * batch_size

            forward_scores, tag_ids = torch.max(temp, 0) # tag_size+2 * batch_size

            tag_ids.masked_fill_(mask_step.expand(padded_tag_size, -1), end_tag)
            back_points.append(tag_ids)

        trace = back_points[-1][end_tag,:]
        decode_idx[-1] = trace
        for idx in xrange(seq_len-2, -1, -1):

            trace = torch.gather(back_points[idx], 0, trace.expand(1, -1))
            decode_idx[idx] = trace

        return decode_idx[1:]


    def forward(self, c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, mask_v, sent_lens, char_lens):
        '''
        mask_v: seq_len * batch_size
        '''

        word_emb = self.dropout.forward(torch.cat([self.word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2))
        if self.ex_word_embeds:
            word_emb2 = self.dropout.forward(torch.cat([self.ex_word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2))
            word_emb = torch.cat([word_emb, word_emb2], dim = 2)
        emission_score, _, _ = self._cal_emission(c_f_v, cf_p_v, c_b_v, cb_p_v, word_emb, sent_lens, char_lens)  # seq_len * batch_size * tag_size

        tag_size = self.tag_size
        seq_len = emission_score.size(0)
        batch_size  = emission_score.size(1)
        start_tag = tag_size
        end_tag = tag_size + 1

        if torch.cuda.is_available():
            neg_mask = autograd.Variable((1-mask_v.data).cuda())
        else:
            neg_mask = autograd.Variable((1-mask_v.data))
        emission_score.masked_fill_(neg_mask.expand(tag_size, -1, -1).permute(1,2,0), 0.0)

        transitions = self.transitions

        transitions.data[:, start_tag] = self.small
        transitions.data[end_tag, :] = self.small
        transitions.data[start_tag, end_tag] = self.small

        b_s = np.array([[self.small] * tag_size + [0.0, self.small]]).astype(np.float32)
        e_s = np.array([[self.small] * tag_size + [self.small, 0.0]]).astype(np.float32)
        if torch.cuda.is_available():
            b_s_v = autograd.Variable(torch.from_numpy(b_s).expand(1, batch_size, -1).cuda()) # batch_size * tag_size+2
            e_s_v = autograd.Variable(torch.from_numpy(e_s).expand(1, batch_size, -1).cuda()) # batch_size * tag_size+2
        else:
            b_s_v = autograd.Variable(torch.from_numpy(b_s).expand(1, batch_size, -1)) # batch_size * tag_size+2
            e_s_v = autograd.Variable(torch.from_numpy(e_s).expand(1, batch_size, -1)) # batch_size * tag_size+2

        s_padding = self.small*torch.ones(seq_len, batch_size, 1)
        e_padding = torch.zeros(seq_len, batch_size, 1)

        # seq_len * batch_size * 2
        if torch.cuda.is_available():
            s_padding_v = autograd.Variable(s_padding.cuda())
            e_padding_v = autograd.Variable(e_padding.cuda())
        else:
            s_padding_v = autograd.Variable(s_padding)
            e_padding_v = autograd.Variable(e_padding)

        e_padding_v.masked_fill_(mask_v.expand(1, -1, -1).permute(1, 2, 0), self.small)

        padded_emission_score = torch.cat([emission_score, s_padding_v, e_padding_v], dim = 2) # seq_len * batch_size * tag_size+2

        unaries = torch.cat([b_s_v, padded_emission_score, e_s_v], dim = 0) # seq_len+2 * batch_size * tag_size+2

        if torch.cuda.is_available():
            to_pad = autograd.Variable(torch.ByteTensor(1, batch_size).fill_(1).cuda())
        else:
            to_pad = autograd.Variable(torch.ByteTensor(1, batch_size).fill_(1))
        padded_mask = torch.cat([to_pad, mask_v], dim = 0)

        tag_seq = self._viterbi_decode(unaries.data, transitions.data, padded_mask.data)

        return tag_seq.cpu()

