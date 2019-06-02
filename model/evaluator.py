
import torch
import numpy as np
import itertools
import sys
import codecs
import os
import pdb

import model.utils as utils
from torch.autograd import Variable

def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            print tags
            raise Exception('Invalid format!')
    return new_tags

def iob_seg(tags):
    new_tags = []

    for i, tag in enumerate(tags):
        tag_seg = tag.split('-')[0]
        '''
        IOB
        '''
        new_tags.append(tag_seg)

    return new_tags

class eval_batch(object):
    """
    Base class for evaluation, provide method to calculate f1 score and accuracy

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
    """

    def __init__(self, c_map, l_map, seg_l_map, ent_l_map, w_map, win_size, gpu=0):
        self.l_map = l_map
        self.seg_l_map = seg_l_map
        self.ent_l_map = ent_l_map

        self.w_map = w_map
        self.c_map = c_map
        self.r_w_map = utils.revlut(w_map)
        self.r_l_map = utils.revlut(l_map)
        self.gpu = gpu
        self.win_size = win_size


    def calc_pred_batch(self, decoded_data, target_data, f_data, len_data, seg_only = False):
        """
        update statics for f1 score

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """

        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)
        batch_words = torch.unbind(f_data, 0)

        predictions = []

        for words, decoded, target, length in zip(batch_words, batch_decoded, batch_targets, len_data):
            gold = target[:length]
            best_path = decoded[:length]
            words_ids = words[:length]

            # '''
            y_reals = [self.r_l_map[idx] for idx in gold.numpy()]
            y_preds = [self.r_l_map[idx] for idx in best_path.numpy()]
            real_words = [self.r_w_map[idx] for idx in words_ids.numpy()]

            # convert iobes to iob
            # '''
            iob_y_reals = iobes_iob([self.r_l_map[idx] for idx in gold.numpy()])

            iob_y_preds = iobes_iob([self.r_l_map[idx] for idx in best_path.numpy()])
            if seg_only:
                iob_y_reals = iob_seg(iob_y_reals)
                iob_y_preds = iob_seg(iob_y_preds)
            # '''

            for i, (word, iob_y_real, iob_y_pred) in enumerate(zip(real_words, iob_y_reals, iob_y_preds)):
                new_line = " ".join([word, iob_y_real, iob_y_pred])
                predictions.append(new_line)
            predictions.append("")

        return predictions


    def f1_score(self, predictions, data_type):
        """
        calculate f1 score based on statics
        """
        eval_script = './conlleval.pl'
        with codecs.open(data_type+'temp_output.txt', 'w', 'utf8') as f:
            f.write("\n".join(predictions))
        os.system("{} < {} > {}".format(eval_script, data_type+'temp_output.txt', data_type+'scores.txt'))

        # CoNLL evaluation results
        eval_lines = [l.rstrip() for l in codecs.open(data_type+'scores.txt', 'r', 'utf8')]

        f1 = float(eval_lines[1].strip().split()[-1])
        recall = float(eval_lines[1].strip().split()[-3].replace('%;', ''))
        precision = float(eval_lines[1].strip().split()[-5].replace('%;', ''))
        accuracy = float(eval_lines[1].strip().split()[-7].replace('%;', ''))

        return f1, precision, recall, accuracy

    def calc_score(self, predictions, data_type):

        f, precision, recall, accuracy = self.f1_score(predictions, data_type)
        return f, precision, recall, accuracy
