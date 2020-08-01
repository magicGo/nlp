# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 12:54
# @Author  : Magic
# @Email   : hanjunm@haier.com
import argparse
import os
import numpy as np

import tensorflow as tf

import modeling
import tokenization
from ner.run_ner import InputFeatures, HaierProcessor

def convert_single_text(text, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    textlist = text.split()

    tokens = []

    for word in textlist:
        token = tokenizer.tokenize(word, type='ner')
        tokens.extend(token)

    if len(tokens) > max_seq_length - 1:
        tokens = tokens[0: (max_seq_length - 1)]

    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    for index, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        ntokens.append('[PAD]')

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(ntokens) == max_seq_length

    tf.logging.info("*** Example ***")
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=None,
        is_real_example=True)
    return feature


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_sequence_output()

    with tf.variable_scope("loss"):
        ln_type = bert_config.ln_type
        if ln_type == 'preln':  # add by brightmart, 10-06. if it is preln, we need to an additonal layer: layer normalization as suggested in paper "ON LAYER NORMALIZATION IN THE TRANSFORMER ARCHITECTURE"
            print("ln_type is preln. add LN layer.")
            output_layer = layer_norm(output_layer)
        else:
            print("ln_type is postln or other,do nothing.")

        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        project_logits = tf.keras.layers.Dense(
            units=num_labels,
            activation=None,
            kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.02),
            bias_initializer=tf.keras.initializers.zeros()
        )(output_layer)

        project_logits = tf.reshape(project_logits, [-1, args.max_seq_len, num_labels])

        mask2len = tf.reduce_sum(input_mask, axis=1)

        trans = tf.get_variable(
            "transitions",
            shape=[num_labels, num_labels],
            initializer=tf.contrib.layers.xavier_initializer())

        predicts, viterbi_score = tf.contrib.crf.crf_decode(project_logits, trans, mask2len)

        predicts = predicts[:, :]

    return predicts


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def predict(args):
    """
    加载中文分类模型
    :param args:
    :param num_labels:
    :param logger:
    :return:
    """
    tf.logging.set_verbosity(tf.logging.INFO)

    processor = HaierProcessor(None, args.model_dir)

    label_map = processor.get_label_map()

    inv_label_map = processor.get_inv_label_map()

    tokenizer = tokenization.FullTokenizer(vocab_file='models/albert_tiny/vocab.txt', do_lower_case=True)

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')
        segment_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'segment_ids')

        bert_config = modeling.BertConfig.from_json_file(os.path.join(args.bert_model_dir, 'albert_config_tiny.json'))

        predicts = create_model(
            bert_config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            labels=None,
            num_labels=len(label_map),
            use_one_hot_embeddings=False)

        saver = tf.train.Saver()

        latest_checkpoint = tf.train.latest_checkpoint(args.model_dir)
        tf.logging.info('loading... %s ' % latest_checkpoint)
        saver.restore(sess, latest_checkpoint)
    return sess, tokenizer, inv_label_map, input_ids, input_mask, segment_ids, predicts


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.bert_model_dir = 'models/albert_tiny'
    args.model_dir = 'output/ner/Dev.sweepingRobot'
    args.max_seq_len = 31

    sess, tokenizer, inv_label_map, input_ids, input_mask, segment_ids, predicts = predict(args)
    while True:
        text = input('exit input `exit`, please input text: ')
        if text == 'exit':
            sess.close()
        feature = convert_single_text(text.strip(), args.max_seq_len, tokenizer)
        feed_dict = {}
        feed_dict[input_ids] = np.asarray([feature.input_ids], dtype=np.int32)
        feed_dict[input_mask] = np.asarray([feature.input_mask], dtype=np.int32)
        feed_dict[segment_ids] = np.asarray([feature.segment_ids], dtype=np.int32)
        prediction = sess.run(fetches=[predicts], feed_dict=feed_dict)
        print(prediction)
        result = []
        for index in range(1, len(text) + 1):
            result.append(inv_label_map[str(prediction[0][0][index])])

        print(result)