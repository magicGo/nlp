# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 18:27
# @Author  : Magic
# @Email   : hanjunm@haier.com
import json
import os
from shutil import rmtree

import tensorflow as tf
import yaml


def freeze_session(session, model, keep_var_names=None, output_names=None, clear_devices=True):
    """
    :param session: 需要转换的tensorflow的session
    :param keep_var_names:需要保留的variable，默认全部转换constant
    :param output_names:output的名字
    :param clear_devices:是否移除设备指令以获得更好的可移植性
    :return:
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        # 如果指定了output名字，则复制一个新的Tensor，并且以指定的名字命名
        if len(output_names) > 0:
            for i in range(len(output_names)):
                # 当前graph中复制一个新的Tensor，指定名字
                tf.identity(model.outputs[i], name=output_names[i])
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def freeze_session2(checkpoint_dir, session, model):
    # 只需要修改这一段，定义输入输出，其他保持默认即可
    inputs = {}
    input_bak = {}
    inputs['input'] = tf.saved_model.utils.build_tensor_info(model.input)
    input_bak['input'] = model.input.op.name

    with tf.gfile.GFile(os.path.join(checkpoint_dir, 'inputs.json'), 'w') as output:
        output.write(json.dumps(input_bak, ensure_ascii=False, indent=4))

    outputs = {}
    output_bak = {}
    outputs['y_prob'] = tf.saved_model.utils.build_tensor_info(model.output)
    output_bak['y_prob'] = model.output.op.name

    with tf.gfile.GFile(os.path.join(checkpoint_dir, 'outputs.json'), 'w') as output:
        output.write(json.dumps(output_bak, ensure_ascii=False, indent=4))

    model_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    export_path = os.path.join(checkpoint_dir, 'model')
    if os.path.exists(export_path):
        rmtree(export_path)
    tf.logging.info("Export the model to {}".format(export_path))

    # try:
    legacy_init_op = tf.group(
        tf.tables_initializer(), name='legacy_init_op')
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        session, [tf.saved_model.tag_constants.SERVING],
        clear_devices=True,
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                model_signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()