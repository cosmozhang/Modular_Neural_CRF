"""

"""

import codecs
import csv
import itertools
from functools import reduce
import sys
from collections import defaultdict
import tensor_utils
from torch import autograd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init

import json
import operator
import pdb

# pdb.set_trace()

zip = getattr(itertools, 'izip', zip)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_user(s):
    if len(s)>1 and s[0] == "@":
        return True
    else:
        return False

def is_url(s):
    if len(s)>4 and s[:5] == "http:":
        return True
    else:
        return False

def tweet_sconvt(s):
    if is_number(s):
        return '<DIGIT>'
    elif is_user(s):
        return '<USR>'
    elif is_url(s):
        return '<URL>'
    else:
        return s

def iob2(tags):
    """
    Check that tags have a valid BIO format.
    Tags in BIO1 format are converted to BIO2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True

def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def transformtags(tags, senti_tags):
    """
    transform the target sentiment tags to only include person and organization

    args:
        tags: the bioes formed tags
        senti_tags: sentiment tags as a list
    return:
        transformed_tags
    """
    transformed_tags = list()
    for tag, senti_tag in zip(tags, senti_tags):
        splitted_tag = tag.split('-')
        if len(splitted_tag) == 2:
            tag_type = splitted_tag[1]
            if splitted_tag[0] == 'B' or splitted_tag[0] == 'S':
                span_senti = senti_tag
        else:
            tag_type = splitted_tag[0]
        if tag_type == "PERSON" or tag_type == "ORGANIZATION":
            transformed_tag = splitted_tag[0] + '-' + span_senti
        else:
            transformed_tag = 'O'
        transformed_tags.append(transformed_tag)
    return transformed_tags

def encode2char(input_lines, char_dict):
    """
    get char representation of lines

    args:
        input_lines (list of strings) : input corpus
        char_dict (dictionary) : char-level dictionary
    return:
        forw_lines
    """
    unk = char_dict['<u>']
    forw_lines = [list(map(lambda m: list(map(lambda t: char_dict.get(t, unk), m)), line)) for line in input_lines]
    return forw_lines


def concatChar(input_lines, char_dict):
    """
    concat char into string

    args:
        input_lines (list of list of char) : input corpus
        char_dict (dictionary) : char-level dictionary
    return:
        forw_lines
    """
    features = [[char_dict[' ']] + list(reduce(lambda x, y: x + [char_dict[' ']] + y, sentence)) + [char_dict['\n']] for sentence in input_lines]
    return features


def encode_word(input_lines, word_dict):
    """
    encode list of strings into word-level representation with unk
    """
    unk = word_dict['<UNK>']
    lines = list(map(lambda t: list(map(lambda m: word_dict.get(m, unk), t)), input_lines))
    return lines


def encode_label(input_lines, label_dict):
    """
    encode list of strings into word-level representation: number
    """
    lines = list(map(lambda t: list(map(lambda m: label_dict[m], t)), input_lines))
    return lines


def generate_charmapping(texts, c_thresholds=0):


    char_count = dict()
    char_map = dict()

    for sent in texts:
        for word in sent:
            for char in word:
                # print tup
                if char not in char_count:
                    char_count[char] = 0
                else:
                    char_count[char] += 1

    for char, count in char_count.iteritems():
        if count > c_thresholds:
            '''
            start from 0
            '''
            char_map[char] = len(char_map)

    '''
    unk for char
    '''
    char_map['<u>'] = len(char_map)
    '''
    concat for char
    '''
    char_map[' '] = len(char_map)
    '''
    eof for char
    '''
    char_map['\n'] = len(char_map)
    return char_map

def generate_mappings(lines, caseless = True, thresholds=1, unknown = '<UNK>'):

    word_map = dict()
    label_map = dict()
    word_count = dict()

    seg_label_map = dict()

    ent_label_map = dict()

    sentence, sentences = [], []
    '''
    read in data
    '''
    for line in lines:
        if not (line.isspace() or (line.startswith("## Tweet"))):
            line = line.rstrip('\n').split()
            sentence.append(line)
        elif len(sentence) > 0:
            sentences.append(sentence)
            sentence = list()

    for snt in sentences:
        for lin in snt:
            raw_word = lin[0].lower() if caseless else lin[0]
            word = tweet_sconvt(raw_word)
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
        tags = [lin[1] for lin in snt]
        senti_tags = [lin[2] for lin in snt]
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in tags)
            raise Exception('Sentences should be given in BIO format! ' + 'Please check sentence %s' % s_str)
        new_tags = iob_iobes(tags)

        transformed_tags = transformtags(new_tags, senti_tags)

        for nt in transformed_tags:
            if nt not in label_map:
                label_map[nt] = len(label_map)

            nseg = nt.split('-')[0]
            '''
            BIO
            '''
            if nseg not in seg_label_map:
                seg_label_map[nseg] = len(seg_label_map)


            if len(nt.split('-')) ==2:
                nent = nt.split('-')[1]
            else:
                nent = nt.split('-')[0]

            '''
            entity and O
            '''
            if nent not in ent_label_map:
                ent_label_map[nent] = len(ent_label_map)

    for word, count in word_count.iteritems():
        if count > thresholds:
            word_map[word] = len(word_map) + 1

    word_map[unknown] = 0
    '''
    inserting <eof>
    '''
    word_map['<eof>'] = len(word_map)

    '''
    pad2id
    '''
    label_map['<pad>'] = len(label_map)
    seg_label_map['<pad>'] = len(seg_label_map)
    ent_label_map['<pad>'] = len(ent_label_map)

    return word_map, label_map, seg_label_map, ent_label_map

def read_corpus(lines, caseless = True):
    """
    convert corpus into words and labels
    """
    words = list()
    labels = list()
    seg_labels = list()
    ent_labels = list()

    sentences = list()
    sentence = list()

    '''
    read in data
    '''
    for line in lines:
        if not (line.isspace() or (line.startswith("## Tweet"))):
            line = line.rstrip('\n').split()
            sentence.append(line)
        elif len(sentence) > 0:
            sentences.append(sentence)
            sentence = list()

    for snt in sentences:
        if len(snt) <= 1:
            # print lefthand_input
            continue
        new_words = list()
        # for lin in snt:
        #     new_words.append(lin[0])
        new_words = [tweet_sconvt(lin[0].lower() if caseless else lin[0]) for lin in snt]
        # new_tags = [lin[-1] for lin in snt]
        tags = [lin[1] for lin in snt]
        senti_tags = [lin[2] for lin in snt]
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in tags)
            raise Exception('Sentences should be given in BIO format! ' + 'Please check sentence %s' % s_str)
        new_tags = iob_iobes(tags)

        transformed_tags = transformtags(new_tags, senti_tags)

        # print transformed_tags

        new_seg_tags = list()
        new_ent_tags = list()

        for nt in transformed_tags:
            nseg = nt.split('-')[0]
            '''
            BIO
            '''
            new_seg_tags.append(nseg)

            '''
            entity and O
            '''
            if len(nt.split('-')) ==2:
                nent = nt.split('-')[1]
            else:
                nent = nt.split('-')[0]
            new_ent_tags.append(nent)
        words.append(new_words)
        labels.append(transformed_tags)
        seg_labels.append(new_seg_tags)
        ent_labels.append(new_ent_tags)

    return words, labels, seg_labels, ent_labels

def load_embedding_wlm(emb_file, word_map, emb_len, delimiter = ' '):
    """
    load embedding, indoc words would be listed before outdoc words

    args:
        emb_file: path to embedding file
        delimiter: delimiter of lines
        word_map: word dictionary
        unk: string for unknown token
        emb_len: dimension of embedding vectors
    """

    num_words = len(word_map)
    embedding_tensor = torch.FloatTensor(num_words, emb_len)
    tensor_utils.init_embedding(embedding_tensor)
    words_in_emb = []
    words_in_emb_cnt = 0
    for line in open(emb_file, 'r'):
        line = line.split(delimiter)

        word  = line[0]
        if word in word_map:
            # pdb.set_trace()
            vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
            embedding_tensor[word_map[word]] = torch.FloatTensor(vector)
            words_in_emb_cnt += 1
            words_in_emb.append(word)

        else:
            continue

    """
    # pdb.set_trace()
    with open("words_in_emb.txt", "w") as f:
        for word in words_in_emb:
            f.write(word+" ")
    """

    print "{} words of the corpus in the embedding file.".format(words_in_emb_cnt)
    return embedding_tensor

def accumulate(iterable, func=operator.add):
    'Return running totals'

    '''
    accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    '''
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total

def raw2num(word_features, input_label, seg_input_label, ent_input_label, label_dict, seg_label_dict, ent_label_dict, char_dict, word_dict, caseless):
    """
    Construct bucket by mean for viterbi decode, word-level and char-level
    """
    '''
    encode and padding
    '''
    char_features = encode2char(word_features, char_dict)
    char_fea_len = [list(map(lambda t: len(t) + 1, f)) for f in char_features]
    '''
    length of every word
    '''
    char_fw_features = concatChar(char_features, char_dict)

    labels = encode_label(input_label, label_dict)
    labels = list(map(lambda t:  list(t), labels))

    seg_labels = encode_label(seg_input_label, seg_label_dict)
    seg_labels = list(map(lambda t:  list(t), seg_labels))

    ent_labels = encode_label(ent_input_label, ent_label_dict)
    ent_labels = list(map(lambda t:  list(t), ent_labels))

    if caseless:
        word_features = list(map(lambda t: list(map(lambda x: x.lower(), t)), word_features))
    word_ids = encode_word(word_features, word_dict)

    data_set = []
    for w_f, c_f, c_l, label, seg_label, ent_label in zip(word_ids, char_fw_features, char_fea_len, labels, seg_labels, ent_labels):
        data_set.append((w_f, c_f, c_l, label,seg_label, ent_label))

    return data_set

def revlut(lut):
    return {v: k for k, v in lut.items()}

def save_checkpoint(state, track_list, filename):
    '''
    save checkpoint
    '''
    with open(filename+'.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename+'.model')

def to_input_variable(batched_tuples, c_map, w_map, l_map, seg_l_map, ent_l_map, gpu=0):


    char_padding, word_padding, label_padding, seg_label_padding, ent_label_padding = c_map['\n'], w_map['<eof>'], l_map['<pad>'], seg_l_map['<pad>'], ent_l_map['<pad>']
    '''
    return tensors
    '''
    buckets = [[], [], [], [], [], [], [], [], [], [], []]
    max_words_len = max(len(s[2]) for s in batched_tuples)
    max_chars_len = max(len(s[1]) for s in batched_tuples)
    for (words_id, chars_id, chars_len, labels_id, seg_labels_id, ent_labels_id) in batched_tuples:
        cur_len = len(chars_len)

        padded_feature = chars_id + [char_padding] * (max_chars_len- len(chars_id))
        padded_feature_len = chars_len + [1] * (max_words_len - len(chars_len))
        padded_feature_len_cum = list(accumulate(padded_feature_len)) # max_words_len
        buckets[0].append(padded_feature) # chars
        buckets[1].append(padded_feature_len_cum) # position to yield emissions for chars
        buckets[2].append(padded_feature[::-1]) # reversed chars
        buckets[3].append([max_chars_len - 1] + [max_chars_len - 1 - tup for tup in padded_feature_len_cum[:-1]]) # backward positions to yield emissions for chars

        words = words_id + [word_padding] * (max_words_len - cur_len)
        buckets[4].append(words) # words

        labels = labels_id + [label_padding] * (max_words_len - cur_len)
        buckets[5].append(labels) # labels

        seg_labels = seg_labels_id + [seg_label_padding] * (max_words_len - cur_len)
        buckets[6].append(seg_labels) # seg_labels

        ent_labels = ent_labels_id + [ent_label_padding] * (max_words_len - cur_len)
        buckets[7].append(ent_labels) # ent_labels

        mask = [1] * len(labels_id) + [0] * (max_words_len - cur_len)
        buckets[8].append(mask)  # has additional start, mask
        # print mask

        buckets[9].append(cur_len)
        buckets[10].append(max_chars_len)

    c_f, cf_p, c_b, cb_p, w_f, tg, stg, etg, mask, sent_lens, char_lens = torch.LongTensor(buckets[0]), torch.LongTensor(buckets[1]), torch.LongTensor(buckets[2]), torch.LongTensor(buckets[3]), torch.LongTensor(buckets[4]), torch.LongTensor(buckets[5]), torch.LongTensor(buckets[6]), torch.LongTensor(buckets[7]), torch.ByteTensor(buckets[8]), np.asarray(buckets[9]).astype(np.long), np.asarray(buckets[10]).astype(np.long)

    perm_idx = sent_lens.argsort()[::-1]
    sent_lens.sort()
    sent_lens = sent_lens[::-1]

    # pdb.set_trace()
    c_f = c_f[perm_idx.tolist()]
    cf_p = cf_p[perm_idx.tolist()]
    c_b = c_b[perm_idx.tolist()]
    cb_p = cb_p[perm_idx.tolist()]
    w_f = w_f[perm_idx.tolist()]
    tg = tg[perm_idx.tolist()]
    stg = stg[perm_idx.tolist()]
    etg = etg[perm_idx.tolist()]
    mask = mask[perm_idx.tolist()]
    char_lens = char_lens[perm_idx]

    if gpu >= 0:
        return (c_f.cuda(), cf_p.cuda(), c_b.cuda(), cb_p.cuda(), w_f.cuda(), tg.cuda(), stg.cuda(), etg.cuda(), mask.cuda(), sent_lens, char_lens)
    else:
        return (c_f, cf_p, c_b, cb_p, w_f, tg, stg, etg, mask, sent_lens, char_lens)

def vwrap(c_f, cf_p, c_b, cb_p, w_f, tg, stg, etg, mask, w_map, window_size, gpu=0):


    word_padding = w_map['<eof>']
    '''
    wrap to torch variables
    '''

    c_f_v = autograd.Variable(c_f.transpose(0, 1))
    cf_p_v = autograd.Variable(cf_p.transpose(0, 1))
    c_b_v = autograd.Variable(c_b.transpose(0, 1))
    cb_p_v = autograd.Variable(cb_p.transpose(0, 1))

    assert window_size%2 == 1, "window_size needs to be odd"
    topad = window_size/2
    bs, leng = w_f.size()
    w_f = w_f.transpose(0, 1) #length, batch_size

    if window_size != 1:
        # pdb.set_trace()
        if gpu >= 0:
            head_pad = torch.LongTensor(topad, bs).fill_(word_padding).cuda()
            tail_pad = torch.LongTensor(topad, bs).fill_(word_padding).cuda()
        else:
            head_pad = torch.LongTensor(topad, bs).fill_(word_padding)
            tail_pad = torch.LongTensor(topad, bs).fill_(word_padding)
        w_f_padded = torch.cat((head_pad, w_f, tail_pad), dim = 0)
        w_f_v = [autograd.Variable(w_f_padded[i:i+leng]) for i in range(window_size)]
        # pdb.set_trace()
    else:
        w_f_v = [autograd.Variable(w_f)]

    tg_v = autograd.Variable(tg.transpose(0, 1)).unsqueeze(2)
    stg_v = autograd.Variable(stg.transpose(0, 1)).unsqueeze(2)
    etg_v = autograd.Variable(etg.transpose(0, 1)).unsqueeze(2)
    mask_v = autograd.Variable(mask.transpose(0, 1)).contiguous()
    # sent_lens_v = autograd.Variable(sent_lens)
    # char_lens_v = autograd.Variable(char_lens)
    return c_f_v, cf_p_v, c_b_v, cb_p_v, w_f_v, tg_v, stg_v, etg_v, mask_v

def batch_slice(data, batch_size, sort=True):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in xrange(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sents = [data[i * batch_size + b] for b in range(cur_batch_size)]

        if sort:
            sent_ids = sorted(range(cur_batch_size), key=lambda sent_id: len(sents[sent_id][0]), reverse=True)
            sorted_sents = [sents[sent_id] for sent_id in sent_ids]

        yield sorted_sents


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """

    buckets = defaultdict(list)
    for each_tuple in data:
        sent = each_tuple[0]
        buckets[len(sent)].append(each_tuple)

    batched_data = []
    for sent_len in buckets:
        tuples = buckets[sent_len]
        if shuffle: np.random.shuffle(tuples)
        batched_data.extend(list(batch_slice(tuples, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch
