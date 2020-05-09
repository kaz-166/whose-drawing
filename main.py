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
    print(os.getcwd() + l[0])
    img = cv2.resize(img, dsize = (IMAGE_SIZE, IMAGE_SIZE))
    # 一列にした後、0-1のfloat値にする
    train_image.append(img.flatten().astype(np.float32)/255.0)
    # ラベルを1-of-k方式で用意する
    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    train_label.append(tmp)
  # numpy形式に変換
  train_image = np.asarray(train_image)
  train_label = np.asarray(train_label)
  f.close()

  # 同じく検証用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
  f = open(FLAGS.test, 'r')
  test_image = []
  test_label = []
  for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(l[0])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    test_image.append(img.flatten().astype(np.float32)/255.0)
    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    test_label.append(tmp)
  test_image = np.asarray(test_image)
  test_label = np.asarray(test_label)
  f.close()

  #TensorBoardのグラフに出力するスコープを指定
  with tf.Graph().as_default():
    # 画像を入れるためのTensor(28*28*3(IMAGE_PIXELS)次元の画像が任意の枚数(None)分はいる)
    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    # ラベルを入れるためのTensor(3(NUM_CLASSES)次元のラベルが任意の枚数(None)分入る)
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    # dropout率を入れる仮のTensor
    keep_prob = tf.placeholder("float")

    # inference()を呼び出してモデルを作る
    logits = train.inference(images_placeholder, keep_prob)
    # loss()を呼び出して損失を計算
    loss_value = train.loss(logits, labels_placeholder)
    # training()を呼び出して訓練して学習モデルのパラメーターを調整する
    train_op = train.training(loss_value, FLAGS.learning_rate)
    # 精度の計算
    acc = train.accuracy(logits, labels_placeholder)

    # 保存の準備
    saver = tf.train.Saver()
    # Sessionの作成(TensorFlowの計算は絶対Sessionの中でやらなきゃだめ)
    sess = tf.Session()
    # 変数の初期化(Sessionを開始したらまず初期化)
    sess.run(tf.initialize_all_variables())
    # TensorBoard表示の設定(TensorBoardの宣言的な?)
    summary_op = tf.summary.merge_all()
    # train_dirでTensorBoardログを出力するpathを指定
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph_def)

    x = np.array([0]*FLAGS.max_steps)
    y = np.array([0.0]*FLAGS.max_steps)

    # 実際にmax_stepの回数だけ訓練の実行していく
    for step in range(FLAGS.max_steps):
      for i in range(int(len(train_image)/FLAGS.batch_size)):
        # batch_size分の画像に対して訓練の実行
        batch = FLAGS.batch_size*i
        # feed_dictでplaceholderに入れるデータを指定する
        sess.run(train_op, feed_dict={
          images_placeholder: train_image[batch:batch+FLAGS.batch_size],
          labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
          keep_prob: 0.5})

      # 1step終わるたびに精度を計算する
      train_accuracy = sess.run(acc, feed_dict={
        images_placeholder: train_image,
        labels_placeholder: train_label,
        keep_prob: 1.0})
      print ("step %d, training accuracy %g"%(step, train_accuracy))
      x[step] = step
      y[step] = 1.0 - train_accuracy

      # 1step終わるたびにTensorBoardに表示する値を追加する
      summary_str = sess.run(summary_op, feed_dict={
        images_placeholder: train_image,
        labels_placeholder: train_label,
        keep_prob: 1.0})
      summary_writer.add_summary(summary_str, step)

  # 訓練が終了したらテストデータに対する精度を表示する
  print ("test accuracy %g"%sess.run(acc, feed_dict={
    images_placeholder: test_image,
    labels_placeholder: test_label,
    keep_prob: 1.0}))

  # データを学習して最終的に出来上がったモデルを保存
  # "model.ckpt"は出力されるファイル名
  save_path = saver.save(sess, "model.ckpt")

  plt.plot(x, y)
  plt.title('Training Error')
  plt.xlabel('steps')
  plt.ylabel('error')
  plt.savefig('log/training_error.png')