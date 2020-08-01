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
                for l in label.split():
                    labels.add(l)
        label_list = list(labels)
        label_list.sort()
        for (i, label) in enumerate(label_list):
            inv_label_map[str(i)] = label
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
                la = []
                ld = []
                for l in label.split():
                    label_id = label_map[l]
                    la.append(l)
                    ld.append(label_id)
                examples.append(InputExample(text=text, label=la, tokens=tokens, label_id=ld))
    rng.shuffle(examples)

    return examples


def tokenizer_single_text(tokenizer, text, max_seq_length):
    text = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_text_to_tokens(text)
    inputs = tf.keras.preprocessing.sequence.pad_sequences([input_ids], max_seq_length)
    return inputs


def create_dataset(examples, max_seq_length, label_map):
    inputs = []
    labels = []
    for example in examples:
        inputs.append(example.tokens)
        label_id = [0 for _ in range(len(label_map))]
        for l in example.label_id:
            label_id[int(l)] = 1
        labels.append(label_id)
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, max_seq_length)
    labels = np.asarray(labels)
    return inputs, labels


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    log_file = os.path.join('log', 'train.log')
    if os.path.isfile(log_file):
        os.remove(log_file)
    handlers = [logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    logging.getLogger('tensorflow').handlers = handlers

    new_train = False
    do_train = False
    do_eval = False
    do_predict = True
    restore = True

    model_name = 'textcnn_model.h5'
    black_technology_name = 'trie.json'

    models = [
        os.path.join('intent', 'Heater'),
        os.path.join('intent', 'Heater-controlled'),
        os.path.join('intent', 'BathRoomMaster'),
        os.path.join('intent', 'Dev.sweepingRobot'),
        os.path.join('intent', 'Sterilizer'),
        os.path.join('intent', 'oven')
    ]
    select = 4
    sign = models[select]

    config = textCnnConfig(
        do_train=do_train,
        do_eval=do_eval,
        do_predict=do_predict,
        sign=sign)

    if new_train:
        tokenization.corpus_process(config.input_file)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=config.vocab_file, pre_embedding_file=config.pre_embedding_file, do_lower_case=config.do_lower_case)

    rng = random.Random(config.random_seed)

    label_map, inv_label_map = create_label_map(config.input_file, config.label_file)

    examples = None
    inputs = None
    labels = None
    if config.do_train or config.do_eval:
        examples = create_data_example(config.input_file, tokenizer, label_map, rng)
        inputs, labels = create_dataset(examples, config.sequence_length, label_map)

    model = textCnnModel(
        config=config,
        vocab_size=len(tokenizer.vocab),
        num_classes=len(label_map),
        restore=restore,
        pre_embedding_table=tokenizer.pre_embedding_table,
        multi_embedding=False
    )

    log_dir = os.path.join(os.path.dirname(config.checkpoint_dir), 'log')

    if config.do_train:
        tf.gfile.MakeDirs(config.checkpoint_dir)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(config.checkpoint_dir, model_name),
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
            # validation_split=0.2,
            validation_data=(inputs, labels),
            callbacks=callbacks
        )
        tf.logging.info('training loss is: {}, training accuracy is: {}, val loss is: {}, val accuracy is: {}, lr is: {}'.format(
            np.mean(history.history['loss']),
            np.mean(history.history['acc']),
            np.mean(history.history['val_loss']),
            np.mean(history.history['val_acc']),
            history.history['lr'],
        ))

        model.load_weights(os.path.join(config.checkpoint_dir, model_name))
        freeze_session2(config.checkpoint_dir, tf.keras.backend.get_session(), model)

    if config.do_eval:
        loss, accuracy = model.evaluate(inputs, labels)
        tf.logging.info('eval loss is: {}, eval accuracy is: {}'.format(loss, accuracy))
        regexes = []

        with tf.gfile.GFile(os.path.join(config.checkpoint_dir, black_technology_name), mode='w') as f:
            for example in examples:
                tokens = tokenizer_single_text(tokenizer, ''.join(example.text), config.sequence_length)
                probability = model.predict(np.array(tokens), batch_size=1)
                pred = []
                for index, prob in enumerate(probability[0]):
                    if prob > 0.5:
                        pred.append(inv_label_map[str(index)])
                pred.sort()
                truth = example.label
                truth.sort()
                if not ' '.join(pred) == ' '.join(truth):
                    regex = dict()
                    regex['text'] = ''.join(example.text)
                    regex['value'] = ' '.join(truth)
                    regexes.append(regex)
            f.write(json.dumps(regexes, ensure_ascii=False, indent=4))

    if config.do_predict:
        # freeze_session2(config.checkpoint_dir, tf.keras.backend.get_session(), model)
        while True:
            text = input("请输入测试句子: ").strip()
            input_x = tokenizer_single_text(tokenizer, text, config.sequence_length)
            print(np.array(input_x))
            probability = model.predict(np.array(input_x), batch_size=1)
            print(probability)
            for index, prob in enumerate(probability[0]):
                if prob > 0.5:
                    print(inv_label_map[str(index)])
                    print(prob)

if __name__ == '__main__':
    main()
