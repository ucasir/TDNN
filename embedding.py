# encoding: utf-8

from itertools import izip
import logging

import numpy as np
import copy
import math

logger = logging.getLogger(__name__)


class H5EmbeddingManager(object):
    def __init__(self, h5_path, mode='disk'):
        self.mode = mode
        import h5py
        f = h5py.File(h5_path, 'r')
        if mode == 'disk':
            self.W = f['embedding']
        elif mode == 'in-memory':
            self.W = f['embedding'][:]
        message = "load mode=%s, embedding data type=%s, shape=%s" % (self.mode, type(self.W), self.W.shape)
        logger.info(message)
        words_flatten = f['words_flatten'][0]
        self.id2word = words_flatten.split('\n')
        assert len(self.id2word) == f.attrs['vocab_len'], "%s != %s" % (len(self.id2word), f.attrs['vocab_len'])
        self.word2id = dict(izip(self.id2word, range(len(self.id2word))))
        del words_flatten

    def __getitem__(self, item):
        item_type = type(item)
        if item_type is str:
            index = self.word2id[item]
            embs = self.W[index]
            return embs
        else:
            raise RuntimeError("don't support type: %s" % type(item))

    def init_word_embedding(self, words, dim_size=300, scale=0.1, mode='google'):
        print('loading word embedding.')
        word2id = self.word2id
        W = self.W
        shape = (len(words), dim_size)
        np.random.seed(len(words))
        # W2V = np.random.uniform(low=-scale, high=scale, size=shape).astype('float32')
        W2V = np.zeros(shape, dtype='float32')
        for i, word in enumerate(words[1:], 1):
            if word in word2id:
                _id = word2id[word]
                vec = W[_id]
                vec /= np.linalg.norm(vec)
            elif word.capitalize() in word2id:
                _id = word2id[word.capitalize()]
                vec = W[_id]
                vec /= np.linalg.norm(vec)
            else:
                vec = np.random.normal(0, 1.0, 300)
                vec = (0.01 * vec).astype('float32')
            W2V[i] = vec[:dim_size]
        return W2V



