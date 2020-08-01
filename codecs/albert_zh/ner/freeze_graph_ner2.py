# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 10:36
# @Author  : Magic
# @Email   : hanjunm@haier.com

'''
BERT模型文件 ckpt转pb 工具
'''

# import contextlib
import json
import os

import modeling
import tensorflow as tf
import argparse

os.chdir(os.path.dirname(os.path.dirname(__file__)))

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


def init_predict_var(path):
    with open(os.path.join(path, 'label.json'), 'r', encoding='utf-8') as f:
        label2id = json.load(f)
        id2label = {value: key for key, value in label2id.items()}
        num_labels = len(label2id)
    print('num_labels:%d' % num_labels)
    return num_labels, label2id, id2label


def optimize_class_model(args):
    """
    加载中文分类模型
    :param args:
    :param num_labels:
    :param logger:
    :return:
    """
    tf.logging.set_verbosity(tf.logging.INFO)

    try:
        # 如果PB文件已经存在则，返回PB文件的路径，否则将模型转化为PB文件，并且返回存储PB文件的路径
        tmp_dir = args.model_dir

        pb_file = os.path.join(tmp_dir, 'albert.pb')
        if os.path.exists(pb_file):
            print('pb_file exits', pb_file)
            return pb_file

        num_labels, label2id, id2label = init_predict_var(tmp_dir)

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
                input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')
                segment_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'segment_ids')
                bert_config = modeling.BertConfig.from_json_file(os.path.join(args.bert_model_dir, 'albert_config_tiny.json'))

                prediction = create_model(
                    bert_config=bert_config,
                    is_training=False,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    labels=None,
                    num_labels=num_labels,
                    use_one_hot_embeddings=False
                )

                saver = tf.train.Saver()
                latest_checkpoint = tf.train.latest_checkpoint(args.model_dir)
                tf.logging.info('loading... %s ' % latest_checkpoint)
                saver.restore(sess, latest_checkpoint)
                tf.logging.info('freeze...')
                from tensorflow.python.framework import graph_util
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [prediction.op.name])
                tf.logging.info('predict cut finished !!!')

        # 存储二进制模型到文件中
        tf.logging.info('write graph to a tmp file: %s' % pb_file)
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return pb_file
    except Exception as e:
        tf.logging.error('fail to optimize the graph! %s' % e, exc_info=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trans ckpt file to .pb file')

    args = parser.parse_args()
    args.bert_model_dir = 'models/albert_tiny'
    args.model_dir = 'output/ner/heater/decreaseTemperature'
    args.max_seq_len = 31

    optimize_class_model(args)