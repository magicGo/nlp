# -*- coding: utf-8 -*-
# @Time    : 2019/6/20 17:26
# @Author  : Magic
# @Email   : hanjunm@haier.com
import codecs
import json
import logging
import os
import random
import sys

import tensorflow as tf

import yaml
import numpy as np

import tokenization
from freeze import freeze_session2
from modeling import textCnnModel, textCnnConfig
from db.mysql import Mysql


class InputExample(object):
    def __init__(self, text, label, tokens, label_id=None):
        self.text = text
        self.label = label
        self.tokens = tokens
        self.label_id = label_id


def create_label_map(input_file, label_file):
    inv_label_map = {}
    if tf.gfile.Exists(label_file):
        with tf.gfile.GFile(label_file, 'r') as reader:
            if label_file.endswith('.yml'):
                inv_label_map = yaml.load(reader, Loader=yaml.FullLoader)
            elif label_file.endswith('.json'):
                inv_label_map = json.load(reader)
    else:
        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        with tf.gfile.GFile(input_file, "r") as reader:
            labels = set()
            while True:
                line = tokenization.convert_to_unicode(reader.readline()).strip()
                if not line:
                    break
                label, text = line.split('\t')
                labels.add(label)
        label_list = list(labels)
        for (i, label) in enumerate(label_list):
            inv_label_map[i] = label
        with tf.gfile.GFile(label_file, 'w') as writer:
            if label_file.endswith('.yml'):
                writer.write(yaml.dump(inv_label_map))
            elif label_file.endswith('.json'):
                writer.write(json.dumps(inv_label_map, ensure_ascii=False, indent=4))

    label_map = {v: k for k, v in inv_label_map.items()}

    return label_map, inv_label_map


def create_data_example(input_file, tokenizer, label_map, rng):

    examples = []
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline()).strip()
            if not line:
                break
            label, text = line.split('\t')
            text = tokenizer.tokenize(text)
            if text and label:
                tokens = tokenizer.convert_text_to_tokens(text)
                label_id = label_map[label]
                examples.append(InputExample(text=text, label=label, tokens=tokens, label_id=label_id))
    rng.shuffle(examples)

    return examples


def tokenizer_single_text(tokenizer, text, max_seq_length):
    text = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_text_to_tokens(text)
    inputs = tf.keras.preprocessing.sequence.pad_sequences([input_ids], max_seq_length)
    return inputs


def create_dataset(examples, max_seq_length):
    inputs = []
    labels = []
    for example in examples:
        inputs.append(example.tokens)
        labels.append(example.label_id)
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, max_seq_length)
    labels = tf.keras.utils.to_categorical(labels, len(set(labels)))

    return inputs, labels


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    log_file = os.path.join('log', 'train.log')
    if os.path.isfile(log_file):
        os.remove(log_file)
    handlers = [logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    logging.getLogger('tensorflow').handlers = handlers

    new_train = True
    do_train = True
    do_eval = True
    do_predict = False
    restore = False

    do_batch_predict = False

    model_name = 'textcnn_model.h5'
    black_technology_name = 'trie.yml'

    models = [
        os.path.join('one', 'category'),
        os.path.join('one', 'domain')
    ]
    select = 0
    sign = models[select]

    input_file = os.path.join('data', sign, 'data.tsv')
    label_file = os.path.join('data', sign, 'label.yml')
    init_checkpoint = os.path.join(sign, 'checkpoint', model_name)
    os.makedirs(os.path.dirname(input_file), exist_ok=True)
    domain = models[select]

    if new_train:
        tokenization.corpus_process(input_file)

    config = textCnnConfig(
        do_train=do_train,
        do_eval=do_eval,
        do_predict=do_predict,
        input_file=input_file,
        label_file=label_file,
        init_checkpoint=init_checkpoint)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=config.vocab_file, pre_embedding_file=config.pre_embedding_file, do_lower_case=config.do_lower_case)

    rng = random.Random(config.random_seed)

    label_map, inv_label_map = create_label_map(config.input_file, config.label_file)

    examples = None
    inputs = None
    labels = None
    if config.do_train or config.do_eval:
        examples = create_data_example(config.input_file, tokenizer, label_map, rng)
        inputs, labels = create_dataset(examples, config.sequence_length)

    model = textCnnModel(
        config=config,
        vocab_size=len(tokenizer.vocab),
        num_classes=len(label_map),
        restore=restore,
        pre_embedding_table=tokenizer.pre_embedding_table,
        multi_embedding=False
    )

    checkpoint_dir = os.path.dirname(config.init_checkpoint)
    log_dir = os.path.join(os.path.dirname(checkpoint_dir), 'log')

    if config.do_train:
        tf.gfile.MakeDirs(checkpoint_dir)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, model_name),
                monitor='val_acc',
                verbose=config.verbose,
                save_best_only=True,
                period=config.period,
                save_weights_only=False
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_acc',
                factor=1e-2,
                patience=10.0,
                min_delta=0,
                cooldown=0,
                min_lr=5e-5
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_acc',
                patience=10,
                verbose=config.verbose
            ),
            # tf.keras.callbacks.TensorBoard(
            #     log_dir=log_dir,
            #     write_images=True,
            #     embeddings_freq=0
            # )
        ]
        tf.keras.backend.set_learning_phase(True)
        history = model.fit(
            inputs,
            labels,
            batch_size=config.train_batch_size,
            epochs=100,
            validation_split=0.2,
            callbacks=callbacks
        )
        tf.logging.info('training loss is: {}, training accuracy is: {}, val loss is: {}, val accuracy is: {}, lr is: {}'.format(
            sum(history.history['loss']),
            sum(history.history['acc']),
            sum(history.history['val_loss']),
            sum(history.history['val_acc']),
            history.history['lr'],
        ))

        model.load_weights(config.init_checkpoint)
        freeze_session2(checkpoint_dir, os.path.dirname(config.input_file), tf.keras.backend.get_session(), model)

    if config.do_eval:
        loss, accuracy = model.evaluate(inputs, labels)
        tf.logging.info('eval loss is: {}, eval accuracy is: {}'.format(loss, accuracy))
        regexes = []

        with tf.gfile.GFile(os.path.join(checkpoint_dir, black_technology_name), mode='w') as f:
            for example in examples:
                tokens = tokenizer_single_text(tokenizer, ''.join(example.text), config.sequence_length)
                y = np.argmax(model.predict(np.array(tokens), batch_size=1))
                if not y == example.label_id:
                    regex = dict()
                    regex['text'] = ''.join(example.text)
                    regex['category'] = example.label
                    regexes.append(regex)
            f.write(yaml.dump_all(regexes, indent=4, allow_unicode=True))

    if config.do_predict:
        print('predict domain is: ' + domain)
        while True:
            text = input("请输入测试句子: ").strip()
            input_x = tokenizer_single_text(tokenizer, text, config.sequence_length)
            print(np.array(input_x))
            y = np.argmax(model.predict(np.array(input_x), batch_size=1))
            print('label is: {}'.format(inv_label_map[y]))


    if do_batch_predict:
        print('predict domain is: ' + domain)
        querys = []
        with codecs.open('data/one/category/yuguang.tsv', 'r', 'utf-8') as f:
            for line in f:
                label, query = line.strip().split('\t')
                query = query.strip()
                label = label.strip()
                input_x = tokenizer_single_text(tokenizer, query, config.sequence_length)
                y = np.argmax(model.predict(np.array(input_x), batch_size=1))
                label_p = inv_label_map[y]
                if not label == label_p:
                    # print('label is: {}, true label is {}'.format(inv_label_map[y], inv_label_map[2]))
                    querys.append(line.strip())

        with codecs.open('data/one/category/yuguang_gai.tsv', 'w', 'utf-8') as f:
            f.write('\n'.join(querys))



if __name__ == '__main__':
    main()
