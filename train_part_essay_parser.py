#coding=utf-8
import numpy as np
from metrics import *
from essay_model import EssayModel
from utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras import backend as K
import tensorflow as tf
import os

cpu_num = int(os.environ.get('CPU_NUM', 1))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


def train(args):
    assert args['valid_prompt_id'] != args['test_prompt_id']

    trainsets, testsets, weights = readASAP(args['test_prompt_id'], args['data_dir'])
    _, validsets, _ = readASAP(args['valid_prompt_id'], args['data_dir'])

    train_arraysets = [np.asarray(x) for x in trainsets]
    test_arraysets = [np.asarray(x) for x in testsets]
    # validate on an entire prompt set instead of the most similar essays
    valid_arraysets = [np.asarray(x) for x in validsets]

    train_x = [train_arraysets[0], train_arraysets[1]]
    train_y = train_arraysets[3]

    valid_x = [valid_arraysets[0], valid_arraysets[1]]
    valid_y = valid_arraysets[3]

    graph = EssayModel(args['hidden_dim'], args['dense_dim'], weights, args['max_words'], args['opt'])
    model = graph.model_feature_parser_only()
    model.summary()

    model_path = "%s_%d_lstm%d_%d_%s.h5" % ("part-essay", args['test_prompt_id'], args['hidden_dim'], args['dense_dim'], args['opt'])
    max_kappa, max_valid_kappa, max_spearman, max_pearson = 0, 0, 0, 0

    for epoch in range(args['epochs']):
        model.fit(train_x, train_y,
                  nb_epoch=1,
                  batch_size=64,
                  shuffle=True
                  )

        low, high = asap_ranges.get(args['valid_prompt_id'])
        valid_pred = model.predict(valid_x)
        valid_pred_scores = [round(x * (high - low) + low) for x in valid_pred]
        valid_kappa = quadratic_weighted_kappa(valid_pred_scores, valid_y)

        low, high = asap_ranges.get(args['test_prompt_id'])
        test_pred = model.predict([test_arraysets[0], test_arraysets[1]])
        test_pred_scores = [round(x * (high - low) + low) for x in test_pred]
        test_kappa = quadratic_weighted_kappa(test_pred_scores, test_arraysets[3])
        test_spearman = spearman(test_pred_scores, test_arraysets[3])
        test_pearson = pearson(test_pred_scores, test_arraysets[3])


        if valid_kappa > max_valid_kappa:
            max_kappa = test_kappa
            max_spearman = test_spearman
            max_pearson = test_pearson
            max_valid_kappa = valid_kappa
            model.save(model_path)

        print  "epoch%s: valid_kappa=%s, test_kappa=%s, max_kappa=%s, max_spearman=%s, max_pearson=%s" \
               % (epoch, valid_kappa, test_kappa, max_kappa, max_spearman, max_pearson)

    print 'Final Max Kappa: %s' % (max_kappa)



if __name__ == "__main__":
    args = {'valid_prompt_id': 2, 'test_prompt_id': 1, 'hidden_dim': 50, 'dense_dim': 50, \
            'opt': 'adagrad', 'max_words': 40, 'epochs': 30, \
            'data_dir': '/your/data/folder/'}
    # rerun the train-test for several times and choose the model with the best kappa on the validation set
    for repeat in range(25):
        train(args)
