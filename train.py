# from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
import numpy as np

from model.simple_lstm import LSTM_TH
from model.lstm_crf import LSTM_CRF
import model.utils as utils
import model.tensor_utils as tensor_utils
from model.evaluator import eval_batch
import model.advtr as advtr

import argparse
import json
import os
import sys
from tqdm import tqdm

# import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with simple lstm')
    parser.add_argument('--rand_embedding', action='store_true', help='random initialize word embedding')
    parser.add_argument('--emb_file', default='../emb/glove.twitter.27B.100d.txt', help='path to pre-trained embedding')
    parser.add_argument('--ex_emb_file', default=None, help='path to additional pre-trained embedding')
    parser.add_argument('--train_file', default='./esp/train.1', help='path to training file')
    parser.add_argument('--test_file', default='./esp/test.1', help='path to test file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--unk', default='<UNK>', help='unknow-token in pre-trained embedding')
    parser.add_argument('--char_hidden', type=int, default=25, help='dimension of char-level layers')
    parser.add_argument('--word_hidden', type=int, default=300, help='dimension of word-level layers')
    parser.add_argument('--win_size', type=int, default=1, help='the word level window size')
    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=200, help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start point of epoch')
    parser.add_argument('--checkpoint', default='./checkpoint/', help='checkpoint path')
    parser.add_argument('--caseless', action='store_true', help='caseless or not')
    parser.add_argument('--char_dim', type=int, default=30, help='dimension of char embedding')
    parser.add_argument('--word_dim', type=int, default=100, help='dimension of word embedding')
    parser.add_argument('--ex_word_dim', type=int, default=0, help='dimension of word embedding')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    parser.add_argument('--no_fine_tune', action='store_true', help='fine tune the diction of word embedding or not')
    parser.add_argument('--seg_loss', type=int, choices=range(3), default=0, help='including segmentation loss in the loss function') # 0: None; 1: joint traning; 2: joint prediction
    parser.add_argument('--ent_loss', type=int, choices=range(3), default=0, help='including entity loss in the loss function') # 0: None; 1: joint traning; 2: joint prediction
    # parser.add_argument('--v_lstm', action='store_true', help='variational lstm')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='clip grad at')
    parser.add_argument('--mini_count', type=float, default=0, help='thresholds to replace rare words with <UNK>')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stop')
    parser.add_argument('--high_way', action='store_true', help='use highway layers')
    parser.add_argument('--least_iters', type=int, default=60, help='at least train how many epochs before stop')
    parser.add_argument('--use_crf', action='store_true', help='Use CRF for inference or not')
    parser.add_argument('--seg_only', action='store_true', help='Predict the segmantations only')
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    print('setting:')
    print(args)

    '''
    load corpus
    '''
    print('loading corpus')
    with codecs.open(args.train_file, 'r', 'utf-8') as f:
        train_lines = f.readlines()

    with codecs.open(args.test_file, 'r', 'utf-8') as f:
        test_lines = f.readlines()

    raw_train_words, raw_train_labels, raw_train_seg_labels, raw_train_ent_labels = utils.read_corpus(train_lines)

    len_train = int(len(raw_train_words) * 0.9)

    train_words, train_labels, train_seg_labels, train_ent_labels = raw_train_words[:len_train], raw_train_labels[:len_train], raw_train_seg_labels[:len_train], raw_train_ent_labels[:len_train]
    dev_words, dev_labels, dev_seg_labels, dev_ent_labels = raw_train_words[len_train:], raw_train_labels[len_train:], raw_train_seg_labels[len_train:], raw_train_ent_labels[len_train:]
    test_words, test_labels, test_seg_labels, test_ent_labels = utils.read_corpus(test_lines)

    '''
    generating str2id mapping
    '''
    print('constructing str to id maps')
    w_map, l_map, seg_l_map, ent_l_map = utils.generate_mappings(train_lines+test_lines, caseless = args.caseless, thresholds = args.mini_count, unknown = args.unk)

    c_map = utils.generate_charmapping(train_words+dev_words+test_words)

    '''
    construct dataset
    '''
    print('constructing dataset')
    train_dataset = utils.raw2num(train_words, train_labels, train_seg_labels, train_ent_labels, l_map, seg_l_map, ent_l_map, c_map, w_map, args.caseless)
    dev_dataset = utils.raw2num(dev_words, dev_labels, dev_seg_labels, dev_ent_labels, l_map, seg_l_map, ent_l_map, c_map, w_map, args.caseless)
    test_dataset = utils.raw2num(test_words, test_labels, test_seg_labels, test_ent_labels, l_map, seg_l_map, ent_l_map, c_map, w_map, args.caseless)

    '''
    build model
    '''
    # print 'building model'
    if args.use_crf:
        print 'building model with CRF'
        ner_model = LSTM_CRF(len(l_map)-1, len(c_map), args.char_dim, args.char_hidden, args.word_dim, args.word_hidden, args.win_size, len(w_map), args.drop_out, segtgt_size = len(seg_l_map)-1, enttgt_size = len(ent_l_map)-1, if_highway=args.high_way, ex_embedding_dim = args.ex_word_dim, segment_loss = args.seg_loss, entity_loss = args.ent_loss)
    else:
        print 'building model w/o CRF'
        ner_model = LSTM_TH(len(l_map)-1, len(c_map), args.char_dim, args.char_hidden, args.word_dim, args.word_hidden, args.win_size, len(w_map), args.drop_out, segtgt_size = len(seg_l_map)-1, enttgt_size = len(ent_l_map)-1, if_highway=args.high_way, ex_embedding_dim = args.ex_word_dim, segment_loss = args.seg_loss, entity_loss = args.ent_loss)

    '''
    load pretrained embedding
    '''
    if not args.rand_embedding:
        print('loading embeddings')
        embedding_tensor = utils.load_embedding_wlm(args.emb_file, w_map, args.word_dim)
        if args.ex_emb_file:
            print('loading extra embeddings')
            embedding_tensor2 = utils.load_embedding_wlm(args.ex_emb_file, w_map, args.ex_word_dim)
    else:
        embedding_tensor = torch.FloatTensor(len(w_map), args.word_dim)
        init_embedding(embedding_tensor)

    print("word embedding size: '{}, {}'".format(len(w_map), args.word_dim))
    if torch.cuda.is_available():
        embedding_tensor = embedding_tensor.cuda()
        if args.ex_emb_file:
            embedding_tensor2 = embedding_tensor2.cuda()
    ner_model.load_word_embedding(embedding_tensor, args.no_fine_tune)
    if args.ex_emb_file:
        ner_model.load_word_embedding(embedding_tensor2, args.no_fine_tune, extra = True)

    if args.gpu >= 0 and torch.cuda.is_available():
        print('Using GPU, device: ' + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(12345)
        ner_model.cuda()

    tot_iters = int(np.ceil(len(train_dataset) / float(args.batch_size)))
    best_f1 = float('-inf')
    best_acc = float('-inf')
    track_list = list()

    epoch_list = range(args.start_epoch, args.start_epoch + args.epoch)
    patience_count = 0

    evaluator = eval_batch(c_map, l_map, seg_l_map, ent_l_map, w_map, args.win_size, args.gpu)

    cur_lr = args.lr
    optimizer = optim.SGD(ner_model.parameters(), lr=cur_lr, momentum=args.momentum)
    for epoch_idx, cur_epoch in enumerate(epoch_list):
        start_time = time.time()
        epoch_loss = 0
        ner_model.train()
        for batched_tuples in tqdm(utils.data_iter(train_dataset, batch_size = args.batch_size), mininterval=2, desc=' ----- epoch: %d, Total iterations: %d, current lr: %f' % (cur_epoch, tot_iters, cur_lr), leave=False, file=sys.stdout):

            c_f, cf_p, c_b, cb_p, w_f, tg, stg, etg, mask, sent_lens, char_lens = utils.to_input_variable(batched_tuples, c_map, w_map, l_map, seg_l_map, ent_l_map, gpu=args.gpu)

            '''
            tensor to torch variables
            '''
            c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, stg_v, etg_v, mask_v= utils.vwrap(c_f, cf_p, c_b, cb_p, w_f, tg, stg, etg, mask, w_map, args.win_size, gpu = args.gpu)

            '''
            training in each step
            '''
            ner_model.zero_grad()

            des_loss, seg_loss, ent_loss, batch_embeds = ner_model.cal_loss(c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, stg_v, etg_v, mask_v, sent_lens, char_lens)
            loss = des_loss + seg_loss + ent_loss
            loss.backward()

            eadv = advtr.cal_adv(batch_embeds, mask_v, epsilon=1.0)
            ner_model.zero_grad()

            des_loss_adv, seg_loss_adv, ent_loss_adv = ner_model.cal_adv_loss(c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, stg_v, etg_v, mask_v, sent_lens, char_lens, eadv)
            loss_adv = des_loss_adv + seg_loss_adv + ent_loss_adv
            loss_adv.backward()
            epoch_loss += tensor_utils.to_scalar(loss_adv)

            nn.utils.clip_grad_norm_(ner_model.parameters(), args.clip_grad)
            optimizer.step()
        epoch_loss /= tot_iters

        '''
        update lr by decaying
        '''
        cur_lr = args.lr / (1 + (cur_epoch + 1) * args.lr_decay)
        tensor_utils.adjust_learning_rate(optimizer, cur_lr)

        '''
        eval on dev set & save check_point
        '''
        eval_predictions = []

        ner_model.eval()

        for batched_tuples in utils.data_iter(dev_dataset, batch_size = 50):

            c_f, cf_p, c_b, cb_p, w_f, tg, stg, etg, mask, sent_lens, char_lens = utils.to_input_variable(batched_tuples, c_map, w_map, l_map, seg_l_map, ent_l_map, gpu = args.gpu)

            c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, stg_v, etg_v, mask_v = utils.vwrap(c_f, cf_p, c_b, cb_p, w_f, tg, stg, etg, mask, w_map, args.win_size, gpu = args.gpu)

            decoded = ner_model.forward(c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, mask_v, sent_lens, char_lens)

            eval_predictions = eval_predictions + evaluator.calc_pred_batch(decoded, tg.cpu(), w_f.cpu(), sent_lens, args.seg_only)

        dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(eval_predictions, 'dev_')

        if dev_f1 > best_f1:
            patience_count = 0
            best_f1 = dev_f1

            '''
            eval on test set
            '''
            eval_predictions = []

            ner_model.eval()

            for batched_tuples in utils.data_iter(test_dataset, batch_size = 50):

                c_f, cf_p, c_b, cb_p, w_f, tg, stg, etg, mask, sent_lens, char_lens = utils.to_input_variable(batched_tuples, c_map, w_map, l_map, seg_l_map, ent_l_map, gpu = args.gpu)

                c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, stg_v, etg_v, mask_v = utils.vwrap(c_f, cf_p, c_b, cb_p, w_f, tg, stg, etg, mask, w_map, args.win_size, gpu = args.gpu)

                decoded = ner_model.forward(c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, mask_v, sent_lens, char_lens)

                eval_predictions = eval_predictions + evaluator.calc_pred_batch(decoded, tg.cpu(), w_f.cpu(), sent_lens, args.seg_only)

            test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(eval_predictions, 'test_')

            track_list.append(
                {'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc, 'test_f1': test_f1,
                 'test_acc': test_acc})

            print '(loss: %.4f, epoch: %d, dev F1 = %.4f, dev acc = %.4f, F1 on test = %.4f, acc on test= %.4f), saving...' % (epoch_loss,
                 cur_epoch,
                 dev_f1,
                 dev_acc,
                 test_f1,
                 test_acc)

            try:
                utils.save_checkpoint({
                    'epoch': cur_epoch,
                    'state_dict': ner_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'w_map': w_map,
                    'l_map': l_map,
                    'c_map': c_map,
                }, {'track_list': track_list,
                    'args': vars(args)
                    }, args.checkpoint + 'lstm_ti')
            except Exception as inst:
                print(inst)

        else:
            patience_count += 1
            print '(loss: %.4f, epoch: %d, dev F1 = %.4f, dev acc = %.4f)' % (epoch_loss,
                   cur_epoch,
                   dev_f1,
                   dev_acc)
            track_list.append({'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc})


        print 'epoch: ' + str(cur_epoch) + '\t in ' + str(args.epoch) + ' take: ' + str(
            time.time() - start_time) + ' s'

        if patience_count >= args.patience and cur_epoch >= args.least_iters:
            break

    '''
    print best
    '''
    print 'dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f dev_acc: %.4f test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f\n' % (dev_f1, dev_rec, dev_pre, dev_acc, test_f1, test_rec, test_pre, test_acc)

    '''
    write best results to file for cross_validation computation
    '''
    with open('cross_validation_record.txt', 'a') as f:
        f.write(' '.join([str(test_f1), str(test_rec), str(test_pre), str(test_acc)]) + '\n')

