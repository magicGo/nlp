# -*- coding:utf-8 -*-

"""
@author: hanjunming
@time: 2019/6/11 21:19
"""
import math
import os

import numpy as np
import tensorflow as tf


class textCnnConfig():
    def __init__(self,
                 input_file=None,
                 label_file=None,
                 vocab_file='data/vocab.txt',
                 output_dir='output',
                 pre_embedding_file=None,
                 embedding_size=64,
                 dropout_rate=0.2,
                 initializer_range=0.1,
                 sequence_length=30,
                 num_filters=128,
                 kernel_sizes=[2, 3, 4, 5],
                 learning_rate=1e-3,
                 activate=None,
                 optimizer=tf.keras.optimizers.Adam,
                 train_batch_size=128,
                 eval_batch_size=32,
                 predict_batch_size=8,
                 random_seed=123456,
                 do_lower_case=True,
                 init_checkpoint=None,
                 do_train=None,
                 do_eval=None,
                 do_predict=None,
                 verbose=1,
                 period=1):
        self.output_dir = output_dir
        self.input_file = input_file
        self.vocab_file = vocab_file
        self.label_file = label_file
        self.pre_embedding_file = pre_embedding_file
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.sequence_length = sequence_length
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.activate = activate and activate or self.gelu
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.predict_batch_size = predict_batch_size
        self.random_seed = random_seed
        self.do_lower_case = do_lower_case
        self.init_checkpoint = os.path.join(self.output_dir, init_checkpoint)
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_predict = do_predict
        self.verbose = verbose
        self.period = period

    def gelu(self, x):
        cdf = 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0)))
        return x * cdf


def textCnnModel(config,
                 vocab_size=None,
                 num_classes=None,
                 restore=False,
                 pre_embedding_table=None,
                 multi_embedding=False):
    input = tf.keras.Input(shape=(config.sequence_length,), dtype='int32', name='input')
    if multi_embedding:
        embedding_constant = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=config.embedding_size / 2,
            input_length=config.sequence_length,
            weights=pre_embedding_table,
            dtype='float32',
            trainable=False
        )(input)
        embedding_variable = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=config.embedding_size / 2,
            embeddings_initializer=tf.keras.initializers.he_uniform(seed=config.random_seed),
            input_length=config.sequence_length,
            dtype='float32',
            trainable=True
        )(input)
        embedding = tf.keras.layers.Concatenate(axis=-1)([embedding_variable, embedding_constant])
    else:
        if pre_embedding_table:
            embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=config.embedding_size,
                input_length=config.sequence_length,
                weights=np.asarray([pre_embedding_table]),
                dtype='float32',
                trainable=True
            )(input)
        else:
            embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=config.embedding_size,
                embeddings_initializer=tf.keras.initializers.he_uniform(seed=config.random_seed),
                input_length=config.sequence_length,
                dtype='float32',
                trainable=True
            )(input)

    pooled_outputs = []
    for i, kernel_size in enumerate(config.kernel_sizes):
        c = tf.keras.layers.Conv1D(
            filters=config.num_filters,
            kernel_size=kernel_size,
            strides=1,
            padding='VALID',
            activation=config.activate,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            bias_initializer=tf.keras.initializers.Constant(config.initializer_range),
            dtype='float32'
        )(embedding)
        p = tf.keras.layers.MaxPool1D(
            pool_size=config.sequence_length - kernel_size + 1,
            strides=1,
            padding='VALID',
            dtype='float32'
        )(c)
        pooled_outputs.append(p)
    pool_output = tf.keras.layers.Concatenate(axis=-1)(pooled_outputs)

    x = tf.keras.layers.Reshape(
        target_shape=(config.num_filters * len(config.kernel_sizes),)
    )(pool_output)

    x = tf.keras.layers.Dropout(rate=config.do_train and config.dropout_rate or 0.0)(x)

    output = tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.softmax,
        kernel_initializer=tf.keras.initializers.he_normal(seed=config.random_seed),
        bias_initializer=tf.keras.initializers.Constant(config.initializer_range),
        kernel_regularizer=tf.keras.regularizers.l2(),
        bias_regularizer=tf.keras.regularizers.l2(),
        dtype='float32',
        name='output'
    )(x)

    model = tf.keras.Model(inputs=[input], outputs=[output])

    model.compile(optimizer='adam', loss=[tf.keras.losses.categorical_crossentropy], metrics=['accuracy'])

    # model.summary()

    if restore and config.init_checkpoint:
        model.load_weights(config.init_checkpoint)

    return model