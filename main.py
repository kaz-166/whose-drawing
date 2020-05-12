#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import export
import learning
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
# 学習用データ
flags.DEFINE_string('train', './train_data.txt', 'File name of train data')
# 検証用データ
flags.DEFINE_string('test', './test_data.txt', 'File name of train data')
# TensorBoardのデータ保存先フォルダ
flags.DEFINE_string('train_dir', './data', 'Directory to put the training data.')


if __name__ == '__main__':
  if sys.argv[1] == "cnn":    # Convolution Neural Networkによる学習を行う
    learning.cnn(FLAGS.train, FLAGS.test)
  elif sys.argv[1] == "svm":  # SVMによる学習を行う
    learning.svm(FLAGS.train, FLAGS.test)
  else:                       # オプション指定なしの場合はCNNで学習を行う
    learning.cnn(FLAGS.train, FLAGS.test)



