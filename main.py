#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv2
import random
import export
import learning
import tensorflow as tf

# Flagはデフォルト値やヘルプ画面の説明文を定数っぽく登録できるTensorFlow組み込み関数
flags = tf.app.flags
FLAGS = flags.FLAGS
# 学習用データ
flags.DEFINE_string('train', './train_data.txt', 'File name of train data')
# 検証用データ
flags.DEFINE_string('test', './test_data.txt', 'File name of train data')
# TensorBoardのデータ保存先フォルダ
flags.DEFINE_string('train_dir', './data', 'Directory to put the training data.')


if __name__ == '__main__':
  #Convolution Neural Networkによる学習
  learning.svm(FLAGS.train, FLAGS.test)



