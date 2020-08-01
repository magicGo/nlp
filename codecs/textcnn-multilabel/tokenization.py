# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 13:58
# @Author  : Magic
# @Email   : hanjunm@haier.com
import codecs
import collections
import os
import random
import re
import time

import gensim
import unicodedata

import six
import tensorflow as tf


class FullTokenizer(object):
    def __init__(self, vocab_file, pre_embedding_file=None, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.pre_embedding_table = self.__load_pre_embedding(pre_embedding_file)

    def __load_pre_embedding(self, file):
        if file:
            model = gensim.models.KeyedVectors.load_word2vec_format(file)
            embedding_table = []
            for k, v in self.vocab.items():
                embedding_table.append(model.get_vector(k).tolist())
            return embedding_table
        else:
            return None

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_text_to_tokens(self, text):
        return convert_by_vocab(self.vocab, text)

    def convert_tokens_to_text(self, tokens):
        return convert_by_vocab(self.inv_vocab, tokens)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.numberConverter = NumberConverter()
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self.numberConverter.chinese_to_arabic_num(text)
        text = zero_digits(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)

            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)

        return ''.join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char) or _is_punctuation(char):
                continue
            if _is_whitespace(char):
                continue
            else:
                output.append(char)
        return ''.join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab.get(item, vocab.get('[UNK]')))
    return output


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


class NumberConverter(object):
    def __init__(self):
        self.patternDict = collections.OrderedDict()
        self.numRegex = NumRegex()
        # 三百二十一
        regex = self.numRegex.numRegex + self.numRegex.encode_unicode('百') + self.numRegex.numRegex + self.numRegex.encode_unicode('十') + self.numRegex.numRegex
        exper = '{0}*100+{1}*10+{2}*1'
        self.patternDict[re.compile(regex)] = exper

        # 三百零一
        regex = self.numRegex.numRegex + self.numRegex.encode_unicode('百') + self.numRegex.encode_unicode('零') + self.numRegex.numRegex
        exper = '{0}*100+{1}*1'
        self.patternDict[re.compile(regex)] = exper

        # 三百二
        regex = self.numRegex.numRegex + self.numRegex.encode_unicode('百') + self.numRegex.numRegex
        exper = '{0}*100+{1}*10'
        self.patternDict[re.compile(regex)] = exper

        # 三百
        regex = self.numRegex.numRegex + self.numRegex.encode_unicode('百')
        exper = '{0}*100'
        self.patternDict[re.compile(regex)] = exper

        # 二十一
        regex = self.numRegex.numRegex + self.numRegex.encode_unicode('十') + self.numRegex.numRegex
        exper = '{0}*10+{1}*1'
        self.patternDict[re.compile(regex)] = exper

        # 二十
        regex = self.numRegex.numRegex + self.numRegex.encode_unicode('十')
        exper = '{0}*10'
        self.patternDict[re.compile(regex)] = exper

        # 十一
        regex = self.numRegex.encode_unicode('十') + self.numRegex.numRegex
        exper = '1*10+{0}*1'
        self.patternDict[re.compile(regex)] = exper

        # 十
        regex = self.numRegex.encode_unicode('十')
        exper = '1*10'
        self.patternDict[re.compile(regex)] = exper

        # 一
        regex = self.numRegex.numRegex
        exper = '{0}*1'
        self.patternDict[re.compile(regex)] = exper

    def chinese_to_arabic_num(self, query):

        for pattern, exper in self.patternDict.items():
            search_flag = True
            while search_flag:
                search = pattern.search(query)
                if search:
                    exper = exper.format(*[self.numRegex.numDict[group] for group in search.groups()])
                    key = search.group()
                    value = eval(exper)
                    query = query.replace(key, str(value))
                else:
                    search_flag = False

        return query


class NumRegex(object):
    def __init__(self):
        self.numDict = dict()
        self.numDict['一'] = '1'
        self.numDict['二'] = '2'
        self.numDict['两'] = '2'
        self.numDict['三'] = '3'
        self.numDict['四'] = '4'
        self.numDict['五'] = '5'
        self.numDict['六'] = '6'
        self.numDict['七'] = '7'
        self.numDict['八'] = '8'
        self.numDict['九'] = '9'

        self.numRegex = '(['
        for s in self.numDict.keys():
            self.numRegex = self.numRegex + self.encode_unicode(s)

        self.numRegex = self.numRegex + '])'


    def encode_unicode(self, gbString):
        unicode_bytes = ''
        for s in gbString:
            hexB = hex(ord(s))[2: ]
            unicode_bytes = unicode_bytes + '\\u' + hexB

        return unicode_bytes


def corpus_process(filename):
    bak_filename = os.path.join(os.path.dirname(filename), 'data_bak.tsv')
    os.path.isfile(bak_filename) and os.remove(bak_filename)
    os.rename(filename, bak_filename)
    with codecs.open(bak_filename, 'r', 'utf-8') as f, codecs.open(filename, 'w', 'utf-8') as fp:
        datas = collections.OrderedDict()
        for line in f:
            if len(line.strip()) > 0:
                lines = line.strip().split('\t')
                if len(lines) == 2:
                    label, text = lines
                    datas[text] = label

        result = [v + '\t' + k for k, v in datas.items()]
        fp.write('\n'.join(result))
