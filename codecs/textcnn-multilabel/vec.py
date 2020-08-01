# -*- coding: utf-8 -*-
# @Time    : 2019/6/24 18:06
# @Author  : Magic
# @Email   : hanjunm@haier.com

import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('data/bert_vec.txt')

print(model.get_vector('我'))
