#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv2
import random
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import train
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 識別ラベルの数(今回は3つ)
NUM_CLASSES = 2
# 学習する時の画像のサイズ(px)
IMAGE_SIZE = 28
# 画像の次元数(28px*28px*3(カラー))
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3


# Flagはデフォルト値やヘルプ画面の説明文を定数っぽく登録できるTensorFlow組み込み関数
flags = tf.app.flags
FLAGS = flags.FLAGS
# 学習用データ
flags.DEFINE_string('train', './train_data.txt', 'File name of train data')
# 検証用データ
flags.DEFINE_string('test', './test_data.txt', 'File name of train data')
# TensorBoardのデータ保存先フォルダ
flags.DEFINE_string('train_dir', './data', 'Directory to put the training data.')
# 学習訓練の試行回数
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
# 1回の学習で何枚の画像を使うか
flags.DEFINE_integer('batch_size', 10, 'Batch size Must divide evenly into the dataset sizes.')
# 学習率、小さすぎると学習が進まないし、大きすぎても誤差が収束しなかったり発散したりしてダメとか。繊細
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')



if __name__ == '__main__':
  # 学習用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
  # ファイルを開く
  f = open(FLAGS.train, 'r')
  # データを入れる配列
  train_image = []
  train_label = []
  for line in f:
    # 改行を除いてスペース区切りにする
    line = line.rstrip()
    l = line.split()
    # データを読み込んで28x28に縮小
    img = cv2.imread(os.getcwd() + l[0])
    img = cv2.resize(img, dsize = (IMAGE_SIZE, IMAGE_SIZE))
    # 一列にした後、0-1のfloat値にする
    train_image.append(img.flatten().astype(np.float32)/255.0)
    train_label.append(int(l[1]))
  # numpy形式に変換
  train_image = np.asarray(train_image)
  train_label = np.asarray(train_label)
  f.close()
  

  #Kerasの学習
  model = Sequential()
  model.add(Dense(200, input_dim=IMAGE_PIXELS))
  model.add(Activation("relu"))
  model.add(Dropout(0.2))

  model.add(Dense(200))
  model.add(Activation("relu"))
  model.add(Dropout(0.2))

  model.add(Dense(2))
  model.add(Activation("softmax"))

# オプティマイザにAdamを使用
  opt = Adam(lr=0.001)
# モデルをコンパイル
  model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
 # 学習を実行。10%はテストに使用。
  Y = to_categorical(train_label, 2)
  history = model.fit(train_image, Y, nb_epoch=1500, batch_size=100, validation_split=0.1, verbose = 0)

  plt.plot(history.history['acc'], color='red')
  plt.title('Training Accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.savefig('log/training_error.png')


    # 同じく検証用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
  f = open(FLAGS.test, 'r')
  test_image = []
  test_label = []
  test_path = []
  for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(os.getcwd() + l[0])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    test_image.append(img.flatten().astype(np.float32)/255.0)
    test_label.append(int(l[1]))
    test_path.append(os.getcwd() + l[0])
  test_image = np.asarray(test_image)
  test_label = np.asarray(test_label)
  test_path = np.asarray(test_path)

  f.close()

result = model.predict_classes(test_image)

sum_accuracy = 0.0
for i in range(test_image.shape[0]):
    print("label:", test_label[i], "result:", result[i])
    if test_label[i] == result[i]:
        sum_accuracy += 1
sum_accuracy /= test_image.shape[0]
print("accuracy: ", sum_accuracy)