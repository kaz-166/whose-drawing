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
from tensorflow.keras.layers import Dense, Input, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.python.keras.utils.vis_utils import plot_model 


# 識別ラベルの数(今回は3つ)
NUM_CLASSES = 4
# 学習する時の画像のサイズ(px)
IMAGE_SIZE = 128
# 画像の次元数(28px*28px*3(カラー))
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

LOG_TRAINING_ACCURACY_GRAPH_PATH = './log/training_accuracy.png'
LOG_TRAINING_LOSS_GRAPH_PATH = './log/training_loss.png'
LOG_TRAINING_MODEL_PATH = './log/model.png'
LABEL_ANNOTATION_PATH = './label_annotation.txt'

TRAINING_OPTIMIZER = "SGD(確率的勾配降下法)"
ACTIVATION_FUNC = "relu"


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
    #train_image.append(img.flatten().astype(np.float32)/255.0)
    train_image.append(img.astype(np.float32)/255.0)
    train_label.append(int(l[1]))
  # numpy形式に変換
  train_image = np.asarray(train_image)
  train_label = np.asarray(train_label)
  f.close()
  

  #Kerasの学習
  model = Sequential()

  model.add(Conv2D(3, kernel_size=3, activation=ACTIVATION_FUNC, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(3, kernel_size=3, activation=ACTIVATION_FUNC, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Activation(ACTIVATION_FUNC))
  model.add(Dropout(0.2))

  model.add(Dense(200))
  model.add(Activation(ACTIVATION_FUNC))
  model.add(Dropout(0.2))

  model.add(Dense(NUM_CLASSES))
  model.add(Activation("softmax"))

# オプティマイザにAdamを使用
  opt = Adam(lr=0.001)
# モデルをコンパイル
  model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
 # 学習を実行。10%はテストに使用。
  Y = to_categorical(train_label, NUM_CLASSES)
  history = model.fit(train_image, Y, nb_epoch=400, batch_size=100, validation_split=0.1)

#Accuracyのグラフプロット
  plt.plot(history.history['acc'], color='red')
  plt.title('Training Accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.savefig(LOG_TRAINING_ACCURACY_GRAPH_PATH)
  plt.figure()
#Lossのグラフプロット
  plt.plot(history.history['loss'], color='green')
  plt.title('Loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.savefig(LOG_TRAINING_LOSS_GRAPH_PATH)

    # 同じく検証用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
  f = open(FLAGS.test, 'r')
  test_image = []
  test_label = []
  test_path = []
  test_image_org = []
  for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(os.getcwd() + l[0])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    test_image_org.append(img)
    test_image.append(img.astype(np.float32)/255.0)
    test_label.append(int(l[1]))
    test_path.append(os.getcwd() + l[0])
  test_image = np.asarray(test_image)
  test_label = np.asarray(test_label)
  test_path = np.asarray(test_path)

  f.close()

result = model.predict_classes(test_image)
result_prob = model.predict_proba(test_image)
sum_accuracy = 0.0
for i in range(test_image.shape[0]):
    print("label:", test_label[i], "result:", result[i], "prob: ", result_prob[i])
    if test_label[i] == result[i]:
        sum_accuracy += 1
sum_accuracy /= test_image.shape[0]
print("accuracy: ", sum_accuracy)

plot_model(model, show_shapes=True, to_file=LOG_TRAINING_MODEL_PATH)

#ファイル出力
f = open(LABEL_ANNOTATION_PATH, mode='r')
label_annotation = []
for line in f:
    # 改行を除いてスペース区切りにする
    line = line.rstrip()
    label_annotation.append(line.split())

path = './log/result.html'
f = open(path, mode='w')
f.write("<h1>Results</><br>")
f.write("<hr><h2>Training Results</h2><hr>")
f.write("学習データ画像枚数: " + str(train_image.shape[0]) + "枚<br>")
f.write("最適化法: " + TRAINING_OPTIMIZER + "<br>")
f.write("活性化関数: " + ACTIVATION_FUNC + "<br>")

f.write("<img src=" + os.getcwd() + LOG_TRAINING_ACCURACY_GRAPH_PATH + " alt=\"\"  height=\"400\"  />")
f.write("<img src=" + os.getcwd() + LOG_TRAINING_LOSS_GRAPH_PATH + " alt=\"\"  height=\"400\"  />")
f.write("<hr><h2>Neural Network Structure</h2><hr>")
f.write("<img src=" + os.getcwd() + LOG_TRAINING_MODEL_PATH + " alt=\"\"  witdh=\"400\"  height=\"800\"  /> <br>")
f.write("<hr><h2>Prediction Results</h2><hr>")
f.write("<b>Prediction accuracy: " + str(int(sum_accuracy*100)) + "[%]</b><br>")

f.write("<table border=\"1\">")
f.write("<tr>\n")
f.write("<td><b>Image</b></td>\n")
f.write("<td><b>Illustrator(Predicted)</b></td>\n")
f.write("<td><b>Illustrator(Answer)</b></td>\n")
f.write("<td><b>Probability</b></td>\n")
f.write("<td><b>Result</b></td>\n")

f.write("</tr>")
for i in range(test_path.shape[0]):
    f.write("<td><img src=" + test_path[i] + " alt=\"\"  height=\"50\" /></td>") 
    #最も確率の高い値を抽出する
    max_prob = 0.0
    max_arg = 0
    for p in range(result_prob[i].shape[0]):
      if max_prob < result_prob[i][p]:
        max_prob = result_prob[i][p]
        max_arg = p
    f.write("<td><centor>" + label_annotation[result[i]][1] + "</centor></td>")
    f.write("<td><centor>" + label_annotation[test_label[i]][1] + "</centor></td>")
    f.write("<td>" + str(int(max_prob*100)) + "％</td>")
    if test_label[i] == result[i]:
        f.write("<td><font color=\"green\"><b>Correct</b></font></td>")
    else:
        f.write("<td><font color=\"red\"><b>Incorrect</b></font></td>")
    f.write("</tr>")
f.write("</table>")
f.close()

