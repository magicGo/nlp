# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 12:54
# @Author  : Magic
# @Email   : hanjunm@haier.com
import argparse
import os
import pickle
import numpy as np

import tensorflow as tf

import modeling
import tokenization
from classifier.run_classifier import InputFeatures, HaierProcessor

os.chdir(os.path.dirname(os.getcwd()))

def convert_single_text(text, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    tokens_a = tokenizer.tokenize(text)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

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
        label_id=0,
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

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

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

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)

    return probabilities


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def init_predict_var(path):
    num_labels = 2
    label2id = None
    id2label = None
    label2id_file = os.path.join(path, 'label2id.pkl')
    if os.path.exists(label2id_file):
        with open(label2id_file, 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
            num_labels = len(label2id.items())
        print('num_labels:%d' % num_labels)
    else:
        print('Can\'t found %s' % label2id_file)
    return num_labels, label2id, id2label


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

        probabilities = create_model(
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
    return sess, tokenizer, inv_label_map, input_ids, input_mask, segment_ids, probabilities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.bert_model_dir = 'models/albert_tiny'
    args.model_dir = 'output/intent'
    args.max_seq_len = 32

    sess, tokenizer, inv_label_map, input_ids, input_mask, segment_ids, probabilities = predict(args)
    while True:
        text = input('exit input `exit`, please input text: ')
        if text == 'exit':
            sess.close()
        feature = convert_single_text(text.strip(), args.max_seq_len, tokenizer)
        feed_dict = {}
        feed_dict[input_ids] = np.asarray([feature.input_ids], dtype=np.int32)
        feed_dict[input_mask] = np.asarray([feature.input_mask], dtype=np.int32)
        feed_dict[segment_ids] = np.asarray([feature.segment_ids], dtype=np.int32)
        probs = sess.run(fetches=[probabilities], feed_dict=feed_dict)
        print(probs)
        pred_ids = np.argmax(probs)
        print(inv_label_map[str(pred_ids)])
        print(np.max(probs))
