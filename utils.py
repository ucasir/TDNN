#coding=utf-8
import codecs
import nltk
from embedding import H5EmbeddingManager
from itertools import izip
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import random
from metrics import *

FUNCTION_WORDS = ['pad', 'unk']

# the set of syntactic tags
PARSER_TAGS = {
    u'pad': 0,
    u'NP': 1,
    u'PRN': 2,
    u'VP': 3,
    u'SBAR': 4,
    u'PP': 5,
    u'VBP': 6,
    u'RRB': 7,
    u'CC': 8,
    u'PRT': 9,
    u'FRAG': 10,
    u'ADVP': 11,
    u'SBARQ': 12,
    u'WHADVP': 13,
    u'SQ': 14,
    u'INTJ': 15,
    u'WHNP': 16,
    u'WHPP': 17,
    u'RB': 18,
    u'ADJP': 19,
    u'JJ': 20,
    u'NN': 21,
    u'VB': 22,
    u'VBG': 23,
    u'NNS': 24,
    u'TMP': 25,
    u'RBR': 26,
    u'VBN': 27,
    u'QP': 28,
    u'UCP': 29,
    u'JJR': 30,
    u'SINV': 31,
    u'IN': 32,
    u'POS': 33,
    u'WHADJP': 34,
    u'DT': 34,
    u'VBD': 35,
    u'CONJP': 36,
    u'NX': 37,
    u'X': 38,
    u'PRP': 39,
    u'RRC': 40,
    u'MD': 41,
    u'TO': 42,
    u'NNP': 43,
    u'WRB': 44,
    u'VBZ': 45,
    u'JJS': 46,
    u'NAC': 47,
    u'LRB': 48,
    u'RP': 49,
    u'CD': 50,
    u'LST': 51,
    u'WP': 52,
    u'NNPS': 53,
    u'WDT': 54,
    u'FW': 55,
    u'SYM': 56,
    u'EX': 57,
    u'PDT': 58,
    u'LS': 59
}

english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!']

asap_ranges = {
    1: (2.0, 12.0),
    2: (1.0, 6.0),
    3: (0.0, 3.0),
    4: (0.0, 3.0),
    5: (0.0, 4.0),
    6: (0.0, 4.0),
    7: (0.0, 30.0),
    8: (0.0, 60.0)
}

def get_embedding_info(embedding_path):
    h5_manager = H5EmbeddingManager(embedding_path, mode='in-memory')
    words = h5_manager.id2word

    words = FUNCTION_WORDS + words
    id2word = [x.decode('utf-8') for x in words]
    word2id = dict(izip(id2word, range(len(id2word))))

    input_dim = len(words)
    weights = np.zeros((input_dim, h5_manager.W.shape[1]), np.float32)
    weights[2:] = h5_manager.W
    weights[1] = np.average(weights, axis=0)
    return id2word, word2id, weights

def get_essay(essayfile, vocab):
    essayset = []
    with codecs.open(essayfile, mode='r', encoding='utf-8') as in_essay:
        for line in in_essay:
            line = line.replace('\n', '').lower()
            sentences = nltk.sent_tokenize(line)
            essay = []
            for sentence in sentences:
                multi_gram = nltk.word_tokenize(sentence)
                words = [vocab.get(x, 1) for x in multi_gram]
                essay.append(words)
            essay1 = pad_sequences(essay, 40, padding='post')
            essayset.append(essay1)
    maxlen = max([len(x) for x in essayset])
    essayset = pad_sequences(essayset, maxlen, padding='post')

    return essayset

# get the syntactic parser tags
def get_parser(parser_file):
    parserset = []
    with codecs.open(parser_file, mode='r', encoding='utf-8') as in_parser:
        for line in in_parser:
            sentences = []
            line = line.replace('\n', '')
            for sentence in line.split("."):
                sen = []
                phrases = sentence.split(",")
                for phrase in phrases:
                    phrase = phrase.strip().replace('\n', '')
                    if len(phrase) == 0:
                        continue
                    phrase_v = [PARSER_TAGS.get(x, 0) for x in phrase.split()]
                    sen.append(phrase_v)
                sen = pad_sequences(sen, 10, padding='post')
                sentences.append(sen)
            sentences = pad_sequences(sentences, 10, padding='post')
            parserset.append(sentences)
    maxlen = max([len(x) for x in parserset])
    parserset = pad_sequences(parserset, maxlen, padding='post')

    return parserset


def read_model_scores(scorefile, idfile):
    scores, ids = [], []
    with codecs.open(scorefile, mode='r', encoding='utf-8') as in_score:
        for line in in_score:
            scores.append(float(line))

    with codecs.open(idfile, mode='r', encoding='utf-8') as in_id:
        for line in in_id:
            ids.append(int(line))

    model_score = []
    for score, id in zip(scores, ids):
        low, high = asap_ranges.get(id)
        score = (score - low) / (high - low)
        model_score.append(score)

    return model_score


def read_human_scores(scorefile):
    scores = []
    with codecs.open(scorefile, mode='r', encoding='utf-8') as in_score:
        for line in in_score:
            scores.append(float(line))

    return scores


def readASAP(prompt_id, data_dir):
    embedding_path = data_dir + '/vectors/glove.6B/glove.6B.50d.txt.h5_pre.h5'
    id2word, word2id, weights = get_embedding_info(embedding_path)

    test_essay = data_dir + "/data/essay/" + str(prompt_id)
    test_parser = data_dir + "/data/nontopic/Tparser" + str(prompt_id) 
    test_score = data_dir + "/data/score/" + str(prompt_id)

    train_essay = data_dir + '/data/twopart/' + str(prompt_id)
    train_score = data_dir + '/data/twopart/s' + str(prompt_id)
    train_parser = data_dir + "/data/twopart/ps" + str(prompt_id)


    train_essay_data = get_essay(train_essay, word2id)
    train_parser_data = get_parser(train_parser)
    train_score_model = read_human_scores(train_score)

    train_zero_essay, train_zero_score, train_zero_parser = [], [], []
    train_one_essay, train_one_score, train_one_parser = [], [], []

    for e, s, p in zip(train_essay_data, train_score_model, train_parser_data):
        if s == 0:
            train_zero_essay.append(e)
            train_zero_score.append(s)
            train_zero_parser.append(p)
        else:
            train_one_essay.append(e)
            train_one_score.append(s)
            train_one_parser.append(p)

    nb_one = len(train_one_essay)
    nb_zero = len(train_zero_essay)

    tr_essay, tr_score, tr_parser = [], [], []
    if nb_one < nb_zero:
        tr_essay.extend(train_zero_essay)
        tr_parser.extend(train_zero_parser)
        tr_score.extend(train_zero_score)

        index = [random.choice(range(len(train_one_essay))) for _ in range(nb_zero)]
        tr_essay.extend([train_one_essay[i] for i in index])
        tr_parser.extend([train_one_parser[i] for i in index])

        for i in range(nb_zero):
            tr_score.append(1)
    else:
        tr_essay.extend(train_one_essay)
        tr_parser.extend(train_one_parser)
        tr_score.extend(train_one_score)

        index = [random.choice(range(len(train_zero_essay))) for _ in range(nb_one)]
        tr_essay.extend([train_zero_essay[i] for i in index])
        tr_parser.extend([train_zero_parser[i] for i in index])

        for i in range(nb_one):
            tr_score.append(0)


    test_essay_data = get_essay(test_essay, word2id)
    test_parser_data = get_parser(test_parser)
    test_score_data = read_human_scores(test_score)

    del word2id, id2word
    return (tr_essay, tr_parser, [], tr_score), \
           (test_essay_data, test_parser_data, [], test_score_data), weights
