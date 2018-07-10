from keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense, \
    Convolution1D, GlobalMaxPooling1D, merge, Dropout, Merge, BatchNormalization, Activation
from keras.engine import Model
from keras import optimizers
from utils import PARSER_TAGS


class EssayModel:
    def __init__(self, hidden_dim, dense_dim, weight, max_words, opt="adam"):
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim

        self.vocab_size, self.embedding_dim = weight.shape
        self.weights = [weight]

        self.filters = 1
        self.filter_len = 1

        self.max_sens = 10
        self.max_words = max_words
        self.nb_feature = 21

        self.opt = opt


    # uses only the semantic network
    def model_essay(self):
        input_words = Input(shape=(self.max_words,), dtype='int32')
        embedding_layer = Embedding(input_dim=self.vocab_size,
                                    output_dim=self.embedding_dim,
                                    weights=self.weights,
                                    trainable=False,
                                    mask_zero=True)(input_words)
        bi_lstm_layer = Bidirectional(LSTM(output_dim=self.hidden_dim, return_sequences=False),
                                      merge_mode='concat')(embedding_layer)

        sentence_model = Model(inputs=input_words, outputs=bi_lstm_layer)

        input_essay = Input(shape=(None, self.max_words), dtype='int32')

        essay_layer = TimeDistributed(sentence_model)(input_essay)

        essay_bilstm_layer = Bidirectional(LSTM(output_dim=self.hidden_dim, return_sequences=False),
                                           merge_mode='concat')(essay_layer)

        bn_merge_layer2 = BatchNormalization()(essay_bilstm_layer)
        merge_dense_layer2 = Dense(self.dense_dim, activation='relu')(bn_merge_layer2)
        score_layer = Dense(1, activation='sigmoid', name='pred_score')(merge_dense_layer2)
        essay_model = Model(inputs=input_essay, outputs=score_layer)

        if self.opt == "adam":
            optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=0, clipvalue=10)
        elif self.opt == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=0, clipvalue=10)
        else:
            optimizer = optimizers.Adagrad(lr=0.01, epsilon=1e-06, clipnorm=0, clipvalue=10)

        essay_model.compile(optimizer=optimizer,
                            loss="mean_squared_error",
                            metrics=['mean_squared_error'])
        return essay_model

    # uses both the semantic and syntactic networks
    def model_feature_parser_only(self):
        input_words = Input(shape=(40,), dtype='int32')
        embedding_layer = Embedding(input_dim=self.vocab_size,
                                    output_dim=self.embedding_dim,
                                    weights=self.weights,
                                    trainable=False,
                                    mask_zero=True)(input_words)
        bi_lstm_layer = Bidirectional(LSTM(output_dim=self.hidden_dim, return_sequences=False),
                                      merge_mode='concat')(embedding_layer)
        sentence_model = Model(inputs=input_words, outputs=bi_lstm_layer)

        input_essay = Input(shape=(None, 40), dtype='int32')
        essay_layer = TimeDistributed(sentence_model)(input_essay)
        essay_bilstm_layer = Bidirectional(LSTM(output_dim=self.hidden_dim, return_sequences=False),
                                           merge_mode='concat')(essay_layer)

        input_ngrams_parser = Input(shape=(10,), dtype='int32')
        embedding_parser_layer = Embedding(input_dim=len(PARSER_TAGS),
                                           output_dim=self.embedding_dim,
                                           trainable=True,
                                           mask_zero=True)(input_ngrams_parser)
        bi_lstm_parser_layer = LSTM(output_dim=self.hidden_dim, return_sequences=False)(embedding_parser_layer)
                                             
        ngrams_parser_model = Model(inputs=input_ngrams_parser, outputs=bi_lstm_parser_layer)

        input_sentences_parser = Input(shape=(10, 10), dtype='int32')
        sentence_parser_layer = TimeDistributed(ngrams_parser_model)(input_sentences_parser)
        sentence_bilstm_parser_layer = LSTM(output_dim=self.hidden_dim, return_sequences=False)(sentence_parser_layer)
                                                     
        sentence_parser_model = Model(inputs=input_sentences_parser, outputs=sentence_bilstm_parser_layer)

        input_essay_parser = Input(shape=(None, 10, 10), dtype='int32', name='essay')
        essay_parser_layer = TimeDistributed(sentence_parser_model)(input_essay_parser)
        essay_bilstm_parser_layer = LSTM(output_dim=self.hidden_dim, return_sequences=False)(essay_parser_layer)
                                                  

        merge_layer = Merge(mode='concat')([essay_bilstm_layer, essay_bilstm_parser_layer])
        bn_merge_layer = BatchNormalization()(merge_layer)

        merge_dense_layer1 = Dense(self.dense_dim, activation='relu')(bn_merge_layer)
        drop1 = Dropout(0.5)(merge_dense_layer1)
        bn_merge_layer2 = BatchNormalization()(drop1)
        merge_dense_layer2 = Dense(self.dense_dim, activation='relu')(bn_merge_layer2)
        score_layer = Dense(1, activation='sigmoid', name='pred_score')(merge_dense_layer2)
        essay_model = Model(inputs=[input_essay, input_essay_parser], outputs=score_layer)

        if self.opt == "adam":
            optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=0, clipvalue=10)
        else:
            optimizer = optimizers.Adagrad(lr=0.01, epsilon=1e-06, clipnorm=0, clipvalue=10)
        essay_model.compile(optimizer=optimizer,
                            loss="mean_squared_error",
                            metrics=['mean_squared_error'])
        return essay_model


