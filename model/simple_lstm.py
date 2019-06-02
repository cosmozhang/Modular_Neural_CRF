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
# from model.hrn import HRNN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
import pdb

class LSTM_TH(nn.Module):

    """
    two-headed lstm
    """

    def __init__(self, tagset_size, char_size, char_dim, char_hidden_dim, embedding_dim, word_hidden_dim, win_size, vocab_size, dropout_ratio, tag_dim = 100, segtgt_size = None, enttgt_size = None, if_highway = False, ex_embedding_dim = None, segment_loss=0, entity_loss=0):

        super(LSTM_TH, self).__init__()

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

        self.forw_char_lstm = nn.LSTM(char_dim, char_hidden_dim, bidirectional=False, dropout=dropout_ratio)
        self.back_char_lstm = nn.LSTM(char_dim, char_hidden_dim, bidirectional=False, dropout=dropout_ratio)
        if not ex_embedding_dim:
            self.word_lstm = nn.LSTM(self.word_dim * self.win_size + char_hidden_dim * 2, self.word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)
        else:
            '''
            use two embeddings
            '''
            self.word_lstm = nn.LSTM((self.word_dim + self.ex_word_dim) * self.win_size + char_hidden_dim * 2, self.word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)

        self.word_embeds = nn.Embedding(vocab_size, self.word_dim)

        if self.ex_word_dim > 0:
            self.ex_word_embeds = nn.Embedding(vocab_size, self.ex_word_dim)
        else:
            self.ex_word_embeds = None

        # pdb.set_trace()

        self.dropout = nn.Dropout(p=dropout_ratio)

        '''
        highway nets
        '''
        if if_highway:
            self.fbchar_highway = highway.hw(2 * char_hidden_dim, dropout_ratio=dropout_ratio)

        self.tag_size = tagset_size
        self.seg_size = segtgt_size
        self.ent_size = enttgt_size


        if self.segment_loss != 2 and self.entity_loss != 2:
            self.hidden2tag = nn.Linear(self.word_hidden_dim, self.tag_size)
        elif self.segment_loss == 2 and self.entity_loss !=2:
            self.hidden2tag = nn.Linear(self.word_hidden_dim*2, self.tag_size)
        elif self.segment_loss != 2 and self.entity_loss ==2:
            self.hidden2tag = nn.Linear(self.word_hidden_dim*2, self.tag_size)
        elif self.segment_loss == 2 and self.entity_loss ==2:
            self.hidden2tag = nn.Linear(self.word_hidden_dim*3, self.tag_size)
            '''
            bilinear layer
            '''
            # self.bilinear = nn.Bilinear(self.word_hidden_dim, self.word_hidden_dim, self.tag_size)

        if self.segment_loss != 0 :
            self.segtgt_size = segtgt_size
            if not ex_embedding_dim:
                self.word_lstm_seg = nn.LSTM(self.word_dim * self.win_size + char_hidden_dim * 2, self.word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)
            else:
                '''
                use two embeddings
                '''
                self.word_lstm_seg = nn.LSTM((self.word_dim + self.ex_word_dim) * self.win_size + char_hidden_dim * 2, self.word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)
            self.hidden2seg = nn.Linear(self.word_hidden_dim, self.segtgt_size)

        if self.entity_loss != 0:
            self.enttgt_size = enttgt_size
            if not ex_embedding_dim:
                self.word_lstm_ent = nn.LSTM(self.word_dim * self.win_size + char_hidden_dim * 2, self.word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)
            else:
                '''
                use two embeddings
                '''
                self.word_lstm_ent = nn.LSTM((self.word_dim + self.ex_word_dim) * self.win_size + char_hidden_dim * 2, self.word_hidden_dim // 2, bidirectional=True, dropout=dropout_ratio)
            self.hidden2ent = nn.Linear(self.word_hidden_dim, self.enttgt_size)

        # '''
        # self.tag_embeddings = nn.Parameter(torch.zeros(self.tag_size+2, self.tag_dim)) #tag_embeddings
        # self.to_tag = nn.Parameter(torch.zeros(self.tag_size+2, self.tag_dim))
        # '''


        # pdb.set_trace()

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
        # self.tag_embeddings.data.zero_()
        # self.to_tag.data.zero_()

    def _cal_emission(self, c_f_v, cf_p_v, c_b_v, cb_p_v, word_emb, sent_lens, char_lens):


        word_seq_length = cf_p_v.size(0)
        batch_size = cf_p_v.size(1)

        # """
        '''
        char embedding layer
        '''
        forw_emb = self.char_embeds.forward(c_f_v)
        back_emb = self.char_embeds.forward(c_b_v)

        '''
        dropout
        '''
        forw_emb = self.dropout.forward(forw_emb)
        d_f_emb_packed_in = pack_padded_sequence(forw_emb, char_lens)

        back_emb = self.dropout.forward(back_emb)
        d_b_emb_packed_in = pack_padded_sequence(back_emb, char_lens)

        '''
        feed the whole sequence to lstm
        '''
        packed_cf_out, _ = self.forw_char_lstm.forward(d_f_emb_packed_in)#seq_len_char * batch * char_hidden_dim
        forw_lstm_out, _ = pad_packed_sequence(packed_cf_out)

        packed_cb_out, _ = self.back_char_lstm.forward(d_b_emb_packed_in)#seq_len_char * batch * char_hidden_dim
        back_lstm_out, _ = pad_packed_sequence(packed_cb_out)

        '''
        select positions
        '''
        cf_p_v = cf_p_v.unsqueeze(2).expand(word_seq_length, batch_size, self.char_hidden_dim)
        # pdb.set_trace()
        select_forw_lstm_out = torch.gather(forw_lstm_out, 0, cf_p_v)

        cb_p_v = cb_p_v.unsqueeze(2).expand(word_seq_length, batch_size, self.char_hidden_dim)
        select_back_lstm_out = torch.gather(back_lstm_out, 0, cb_p_v)

        if self.if_highway:
            fb_char_lstm_out = self.dropout(torch.cat((select_forw_lstm_out, select_back_lstm_out), dim=2))
            char_out = self.fbchar_highway.forward(fb_char_lstm_out)
        else:
            char_out = torch.cat((select_forw_lstm_out, select_back_lstm_out), dim=2)

        char_out = self.dropout.forward(char_out)
        # '''
        # pdb.set_trace()

        '''
        combine char and word embeddings
        '''
        word_input = torch.cat((word_emb, char_out), dim = 2)
        # """

        packed_word_input = pack_padded_sequence(word_input, sent_lens)

        #word level lstm
        packed_lstm_out, _ = self.word_lstm.forward(packed_word_input)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out)


        lstm_out = self.dropout.forward(lstm_out)

        if self.segment_loss != 0:

            packed_lstm_out_seg, _ = self.word_lstm_seg.forward(packed_word_input)
            lstm_out_seg, _ = pad_packed_sequence(packed_lstm_out_seg)
            seg_scores_out = self.hidden2seg.forward(lstm_out_seg)
        else:
            seg_scores_out = None

        if self.entity_loss != 0:
            packed_lstm_out_ent, _ = self.word_lstm_ent.forward(packed_word_input)
            lstm_out_ent, _ = pad_packed_sequence(packed_lstm_out_ent)
            ent_scores_out = self.hidden2ent.forward(lstm_out_ent)
        else:
            ent_scores_out = None

        if self.segment_loss == 2 and self.entity_loss == 2:
            # lstm_out = torch.cat((lstm_out, lstm_out_seg, lstm_out_ent), dim = 2)
            lstm_out_seg = torch.sigmoid(torch.matmul(lstm_out, self.W1) + self.b1) * lstm_out_seg
            lstm_out_ent = torch.sigmoid(torch.matmul(lstm_out, self.W2) + self.b2) * lstm_out_ent
            lstm_out = torch.cat((lstm_out_seg, lstm_out, lstm_out_ent), dim = 2)
            # pdb.set_trace()
            # scores_out = self.bilinear.forward(lstm_out_seg, lstm_out_ent)


        '''
        if self.entity_loss == 2:
            lstm_out = torch.cat((lstm_out, ent_scores_out), dim = 2)
        '''

        scores_out = self.hidden2tag.forward(lstm_out) # seq_len * batch_size * tag_size

        return scores_out, F.relu(seg_scores_out) if self.segment_loss != 0 else None, F.relu(ent_scores_out) if self.entity_loss != 0 else None

    def cal_loss(self, c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, segment_target, entity_target, mask_v, sent_lens, char_lens, average_batch = True):

        '''
        tg_v: seq_len * batch_size
        mask_v: seq_len * batch_size
        '''
        batch_size = entity_target.size(1)
        tag_size = self.tag_size
        seg_size = self.seg_size
        ent_size = self.ent_size
        seq_len = entity_target.size(0)

        '''
        word embeddings
        '''
        word_emb = torch.cat([self.word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2)
        if self.ex_word_embeds:
            word_emb2 = torch.cat([self.ex_word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2)
            word_emb = torch.cat([word_emb, word_emb2], dim = 2)
        word_emb.retain_grad()
        # pdb.set_trace()

        emission_score, emission_seg_score, emission_ent_score = self._cal_emission(c_f_v, cf_p_v, c_b_v, cb_p_v, word_emb, sent_lens, char_lens) # seq_len * batch_size * tag_size

        # pdb.set_trace()
        if type(tg_v) == torch.Tensor:
            mask_target = torch.masked_select(tg_v.squeeze(), mask_v.squeeze())
            masked_scores = torch.masked_select(emission_score, mask_v.expand(tag_size,-1,-1).permute(1,2,0)).view(-1, tag_size)
            des_cost = self.xentropy.forward(masked_scores, mask_target)
        else:
            des_cost = 0.0

        if self.segment_loss != 0 and type(segment_target) == torch.Tensor:
            mask_starget = torch.masked_select(segment_target.squeeze(), mask_v.squeeze())
            masked_seg_scores = torch.masked_select(emission_seg_score, mask_v.expand(seg_size,-1,-1).permute(1,2,0)).view(-1, seg_size)
            seg_loss = self.xentropy.forward(masked_seg_scores, mask_starget)
            # cost += seg_loss
        else:
            seg_loss = 0.0

        if self.entity_loss != 0 and type(entity_target) == torch.Tensor:
            mask_etarget = torch.masked_select(entity_target.squeeze(), mask_v.squeeze())
            masked_ent_scores = torch.masked_select(emission_ent_score, mask_v.expand(ent_size,-1,-1).permute(1,2,0)).view(-1, ent_size)
            ent_loss = self.xentropy.forward(masked_ent_scores, mask_etarget)
            # cost += ent_loss
        else:
            ent_loss = 0.0

        if average_batch:
            des_cost = des_cost/ batch_size # average over batch
            seg_loss = seg_loss/ batch_size
            ent_loss = ent_loss/ batch_size

        # print 'wrong'
        return des_cost, seg_loss, ent_loss, word_emb

    def cal_adv_loss(self, c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, segment_target, entity_target, mask_v, sent_lens, char_lens, eadv, average_batch = True):

        '''
        tg_v: seq_len * batch_size
        mask_v: seq_len * batch_size
        '''
        batch_size = entity_target.size(1)
        tag_size = self.tag_size
        seg_size = self.seg_size
        ent_size = self.ent_size
        seq_len = entity_target.size(0)

        # pdb.set_trace()

        #word
        word_emb = torch.cat([self.word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2)
        if self.ex_word_embeds:
            word_emb2 = torch.cat([self.ex_word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2)
            word_emb = torch.cat([word_emb, word_emb2], dim = 2)
        word_emb += eadv
        # pdb.set_trace()
        emission_score, emission_seg_score, emission_ent_score = self._cal_emission(c_f_v, cf_p_v, c_b_v, cb_p_v, word_emb, sent_lens, char_lens) # seq_len * batch_size * tag_size

        # pdb.set_trace()
        if type(tg_v) == torch.Tensor:
            mask_target = torch.masked_select(tg_v.squeeze(), mask_v.squeeze())
            masked_scores = torch.masked_select(emission_score, mask_v.expand(tag_size,-1,-1).permute(1,2,0)).view(-1, tag_size)
            des_cost = self.xentropy.forward(masked_scores, mask_target)
        else:
            des_cost = 0.0

        if self.segment_loss != 0 and type(segment_target) == torch.Tensor:
            mask_starget = torch.masked_select(segment_target.squeeze(), mask_v.squeeze())
            masked_seg_scores = torch.masked_select(emission_seg_score, mask_v.expand(seg_size,-1,-1).permute(1,2,0)).view(-1, seg_size)
            seg_loss = self.xentropy.forward(masked_seg_scores, mask_starget)
            # cost += seg_loss
        else:
            seg_loss = 0.0

        if self.entity_loss != 0 and type(entity_target) == torch.Tensor:
            mask_etarget = torch.masked_select(entity_target.squeeze(), mask_v.squeeze())
            masked_ent_scores = torch.masked_select(emission_ent_score, mask_v.expand(ent_size,-1,-1).permute(1,2,0)).view(-1, ent_size)

            ent_loss = self.xentropy.forward(masked_ent_scores, mask_etarget)
            # cost += ent_loss
        else:
            ent_loss = 0.0

        if average_batch:
            des_cost = des_cost/ batch_size # average over batch
            seg_loss = seg_loss/ batch_size
            ent_loss = ent_loss/ batch_size

        return des_cost, seg_loss, ent_loss

    def forward(self, c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, mask_v, sent_lens, char_lens):
        '''
        mask_v: seq_len * batch_size
        '''

        word_emb = torch.cat([self.word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2)
        if self.ex_word_embeds:
            word_emb2 = torch.cat([self.ex_word_embeds.forward(w_f_v_s) for w_f_v_s in w_f_v], dim = 2)
            word_emb = torch.cat([word_emb, word_emb2], dim = 2)

        emission_score, _, _ = self._cal_emission(c_f_v, cf_p_v, c_b_v, cb_p_v, word_emb, sent_lens, char_lens) # seq_len * batch_size * tag_size

        tag_size = self.tag_size
        batch_size  = emission_score.size(1)
        seq_len = emission_score.size(0)

        if torch.cuda.is_available():
            neg_mask = autograd.Variable((1-mask_v.data).cuda())
        else:
            neg_mask = autograd.Variable((1-mask_v.data))
        emission_score.masked_fill_(neg_mask.expand(tag_size, -1, -1).permute(1,2,0), 0.0)



        vs, tag_seq = torch.max(emission_score.data, 2)
        # print inds.size()
        # return inds.cpu()

        return tag_seq.cpu()

