# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import json
import logging
import math
import os
from random import shuffle

import modeling
import optimization_finetuning as optimization
import tf_metrics
import tokenization
import tensorflow as tf

from early_stopping import stop_if_no_increase_hook

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", 'data/ner/Heater-controlled',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", 'models/albert_tiny/albert_config_tiny.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'haier', "The name of the task to train.")

flags.DEFINE_string("vocab_file", 'models/albert_tiny/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", 'output/ner/Heater-controlled',
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", 'models/albert_tiny/albert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 31,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("batch_size", 32, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 6e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", None,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", None,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_predict_examples(self):
        """See base class."""
        raise NotImplementedError()

    def get_label_map(self):
        """See base class."""
        return NotImplementedError()

    def get_inv_label_map(self):
        """See base class."""
        return NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # tokenization.ner_corpus_process(input_file)
        with tf.gfile.Open(input_file, "r") as f:
            lines = []
            for line in f:
                lines.append(line.strip().split('[SEP]'))
            return lines


class HaierProcessor(DataProcessor):
    """Processor for the my data set."""

    def __init__(self, data_dir, output_dir, do_train=None, do_eval=None, do_predict=None):
        self.output_dir = output_dir
        if data_dir:
            if do_train:
                self.data = self._read_tsv(os.path.join(data_dir, "data.tsv"))
                self.train_examples = self._create_examples(self.data, "train")
                self.dev_examples = self._create_examples(self.data, "dev")

            if do_eval:
                if os.path.isfile(os.path.join(data_dir, "test_data.tsv")):
                    self.test_data = self._read_tsv(os.path.join(data_dir, "test_data.tsv"))
                else:
                    self.test_data = self._read_tsv(os.path.join(data_dir, "data.tsv"))
                self.test_examples = self._create_examples(self.test_data, "test")

            if do_predict:
                self.predict_examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "data.tsv")), "predict")

        if tf.gfile.Exists(os.path.join(self.output_dir, 'label.json')):
            with tf.gfile.GFile(os.path.join(self.output_dir, 'label.json'), 'r') as f:
                self.inv_label_map = json.load(f)
                self.label_map = {label: int(index) for index, label in self.inv_label_map.items()}

    def get_train_examples(self):
        """See base class."""
        return self.train_examples

    def get_dev_examples(self):
        """See base class."""
        return self.dev_examples

    def get_test_examples(self):
        """See base class."""
        return self.test_examples

    def get_predict_examples(self):
        """See base class."""
        return self.predict_examples

    def get_label_map(self):
        """See base class."""
        return self.label_map

    def get_inv_label_map(self):
        """See base class."""
        return self.inv_label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = set()
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0].strip()
            label = line[1].strip()
            if set_type == 'train':
                for l in label.split():
                    labels.add(l)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        shuffle(examples)

        if tf.gfile.Exists(os.path.join(self.output_dir, 'label.json')):
            with tf.gfile.GFile(os.path.join(self.output_dir, 'label.json'), 'r') as f:
                self.inv_label_map = json.load(f)
                self.label_map = {label: int(index) for index, label in self.inv_label_map.items()}
        else:
            if set_type == 'train':
                labels.add('[CLS]')
                labels = list(labels)
                labels.sort()
                self.label_map = {label: index for index, label in enumerate(labels)}
                self.inv_label_map = {index: label for label, index in self.label_map.items()}
                with tf.gfile.GFile(os.path.join(self.output_dir, 'label.json'), 'w') as f:
                    f.write(json.dumps(self.inv_label_map, ensure_ascii=False, indent=4))

        return examples


def convert_single_example(ex_index, example, label_map, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * max_seq_length,
            is_real_example=False)

    textlist = example.text_a.split()
    labellist = example.label.split()

    tokens = []
    labels = []

    for i, (word, label) in enumerate(zip(textlist, labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        labels.append(label)

    if len(tokens) > max_seq_length - 1:
        tokens = tokens[0: (max_seq_length - 1)]
        labels = labels[0: (max_seq_length - 1)]

    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for index, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[index]])

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append('[PAD]')

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length

    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    if tf.gfile.Exists(output_file):
        return

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_map,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


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
    output_layer = model.get_sequence_output()

    with tf.variable_scope("loss"):
        ln_type = bert_config.ln_type
        if ln_type == 'preln':  # add by brightmart, 10-06. if it is preln, we need to an additonal layer: layer normalization as suggested in paper "ON LAYER NORMALIZATION IN THE TRANSFORMER ARCHITECTURE"
            tf.logging.info("ln_type is preln. add LN layer.")
            output_layer = layer_norm(output_layer)
        else:
            tf.logging.info("ln_type is postln or other,do nothing.")

        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        project_logits = tf.keras.layers.Dense(
            units=num_labels,
            activation=None,
            kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.02),
            bias_initializer=tf.keras.initializers.zeros()
        )(output_layer)

        project_logits = tf.reshape(project_logits, [-1, FLAGS.max_seq_length, num_labels])

        mask2len = tf.reduce_sum(input_mask, axis=1)

        trans = tf.get_variable(
            "transitions",
            shape=[num_labels, num_labels],
            initializer=tf.contrib.layers.xavier_initializer())
        log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
            inputs=project_logits,
            tag_indices=labels,
            transition_params=trans,
            sequence_lengths=mask2len)

        loss = tf.reduce_mean(-log_likelihood)

        predicts, viterbi_score = tf.contrib.crf.crf_decode(project_logits, trans, mask2len)

        predicts = predicts[:,: ]

    return loss, project_logits, predicts, trans, mask2len


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits, predicts, trans, mask2len) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s" % (var.name, var.shape, init_string))

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op
            )
            tf.summary.scalar('loss', total_loss)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(label_ids, predictions, num_labels, input_mask, is_real_example):
                accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions, weights=is_real_example)
                # precision, precision_update_op = tf.metrics.precision(labels=label_ids, predictions=predictions)
                # recall, recall_update_op = tf.metrics.recall(labels=label_ids, predictions=predictions)
                precision = tf_metrics.precision(label_ids, predictions, num_labels, average="macro")
                recall = tf_metrics.recall(label_ids, predictions, num_labels, average="macro")
                f1_score = tf_metrics.f1(label_ids, predictions, num_labels, average="macro")
                return {
                    "eval_accuracy": accuracy,
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f1_score": f1_score
                }
                # return {
                #     "eval_accuracy": accuracy,
                #     "eval_precision": (precision, precision_update_op),
                #     "eval_recall": (recall, recall_update_op)
                # }

            eval_metrics = metric_fn(
                label_ids,
                predicts,
                num_labels,
                input_mask,
                is_real_example
            )
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"predictions": predicts})
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn

# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    tf.logging.set_verbosity(tf.logging.INFO)
    nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    handlers = [
        logging.FileHandler(os.path.join(FLAGS.output_dir, nowTime + '.log'), encoding='utf-8')
                ]
    logging.getLogger('tensorflow').handlers = handlers

    processors = {
        "haier": HaierProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](FLAGS.data_dir, FLAGS.output_dir, FLAGS.do_train, FLAGS.do_eval, FLAGS.do_predict)

    label_map = processor.get_label_map()
    label_list = list(label_map.keys())
    label_list.sort()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tf.logging.info("load estimator ...")

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        gpu_options={"allow_growth": True}
    )

    train_examples = None
    dev_examples = None
    num_train_steps = None
    num_warmup_steps = None
    save_checkpoints_steps = None
    run_config = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples()
        tf.logging.info("###length of total train_examples: %d" % len(train_examples))
        save_checkpoints_steps = math.ceil(len(train_examples) / FLAGS.batch_size)
        run_config = tf.estimator.RunConfig(
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            keep_checkpoint_max=25,
            session_config=None
        )
        num_train_steps = int(len(train_examples) / FLAGS.batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        dev_examples = processor.get_dev_examples()

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
            "batch_size": FLAGS.batch_size
        }
    )

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        train_file_exists = os.path.exists(train_file)
        tf.logging.info("###train_file_exists: %s;train_file: %s" % (train_file_exists, train_file))
        if not train_file_exists:  # if tf_record file not exist, convert from raw text file. # TODO
            file_based_convert_examples_to_features(train_examples, label_map, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)

        eval_file = os.path.join(FLAGS.output_dir, "dev.tf_record")
        file_based_convert_examples_to_features(dev_examples, label_map, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(dev_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        early_stopping_hook = stop_if_no_increase_hook(
            estimator=estimator,
            metric_name='eval_f1_score',
            max_steps_without_increase=save_checkpoints_steps * 10,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=save_checkpoints_steps)

        logging_hook = tf.train.LoggingTensorHook(tensors={"train loss": "loss/Mean:0"}, every_n_iter=1)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[early_stopping_hook, logging_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=0)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.do_eval:
        test_examples = processor.get_test_examples()
        num_actual_test_examples = len(test_examples)

        eval_file = os.path.join(FLAGS.output_dir, "test.tf_record")
        file_based_convert_examples_to_features(
            test_examples, label_map, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(test_examples), num_actual_test_examples,
                        len(test_examples) - num_actual_test_examples)
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        #######################################################################################################################
        # evaluate all checkpoints; you can use the checkpoint with the best dev accuarcy

        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=latest_checkpoint)
        tf.logging.info("***** Eval results %s *****" % (latest_checkpoint))
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))

        #######################################################################################################################

    if FLAGS.do_predict:
        predict_examples = processor.get_predict_examples()
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_map,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)

        result = estimator.predict(input_fn=predict_input_fn, checkpoint_path=latest_checkpoint)

        trie_file = os.path.join(FLAGS.output_dir, "trie.json")
        regexes = []
        with tf.gfile.GFile(trie_file, "w") as fp:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                predictions = prediction["predictions"]
                text = predict_examples[i].text_a
                t_label = predict_examples[i].label
                p_label = ' '.join([processor.get_inv_label_map()[str(predict)] for predict in predictions][1: len(text.split()) + 1])
                if i >= num_actual_predict_examples:
                    break
                if t_label != p_label:
                    print(t_label)
                    print(p_label)
                    regex = dict()
                    regex['text'] = ''.join(tokenizer.tokenize(text))
                    regex['value'] = t_label
                    regexes.append(regex)
                num_written_lines += 1
            fp.write(json.dumps(regexes, ensure_ascii=False, indent=4))
            # fp.write(yaml.dump_all(regexes, indent=4, allow_unicode=True))
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    tf.app.run()
