#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
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
import imgproc
import export
from sklearn import svm as sksvm
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model

#Convolutional Neural Networkによる学習
def cnn(train_dir, test_dir):
  # 識別ラベルの数(今回は3つ)
  NUM_CLASSES = 4
  # 学習する時の画像のサイズ(px)
  IMAGE_SIZE = 128
  # 画像の次元数(28px*28px*3(カラー))
  IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

  LABEL_ANNOTATION_PATH = './label_annotation.txt'
  LOG_TRAINING_ACCURACY_GRAPH_PATH = './log/cnn/training_accuracy.png'
  LOG_TRAINING_LOSS_GRAPH_PATH = './log/cnn/training_loss.png'
  LOG_TRAINING_MODEL_PATH = './log/cnn/model.png'
  TRAINING_OPTIMIZER = "SGD(確率的勾配降下法)"
  ACTIVATION_FUNC = "relu"
 # 学習用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
  # ファイルを開く
  f = open(train_dir, 'r')
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

  model.add(Dense(200))
  model.add(Activation(ACTIVATION_FUNC))
  model.add(Dropout(0.2))

  model.add(Dense(200))
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

  export.plot(history)

    # 同じく検証用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
  f = open(test_dir, 'r')
  test_image = []
  test_label = []
  test_path = []
  for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(os.getcwd() + l[0])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
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

  #中間層の出力
  layer_name = 'conv2d'
  #中間層のmodelを作成
  intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
  #出力をmodel.predictで見る
  intermediate_output = intermediate_layer_model.predict(test_image)
  path = os.getcwd() + "/log/cnn/" + layer_name
  if os.path.exists(path) == False:
    os.mkdir(path)
  for i in range(intermediate_output.shape[0]):
    cv2.imwrite(path + '/immidiate_' + str(i) +'.png', intermediate_output[i]*255)

  layer_name = 'conv2d_1'
  #中間層のmodelを作成
  intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
  path = os.getcwd() + "/log/cnn/" + layer_name
  if os.path.exists(path) == False:
      os.mkdir(path)
  #出力をmodel.predictで見る
  intermediate_output = intermediate_layer_model.predict(test_image)
  for i in range(intermediate_output.shape[0]):
    cv2.imwrite(path + '/immidiate_' + str(i) +'.png', intermediate_output[i]*255)

  #結果をhtmlファイル出力
  result_dict = {'acc':0, 'n_img':0, 'opt':"", 'act_func':""}
  result_dict['acc'] = sum_accuracy
  result_dict['n_img'] = train_image.shape[0]
  result_dict['opt'] = TRAINING_OPTIMIZER
  result_dict['act_func'] = ACTIVATION_FUNC
  export.cnn_html(result_dict, test_image, test_label, test_path, result, result_prob)

#SVMによる学習
def svm(train_dir, test_dir):
  # 識別ラベルの数(今回は3つ)
  NUM_CLASSES = 4
  # 学習する時の画像のサイズ(px)
  IMAGE_SIZE = 128
  # 画像の次元数(28px*28px*3(カラー))
  IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3
  # 学習用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
  f = open(train_dir, 'r')
  # データを入れる配列
  train_image = []
  train_label = []
  train_features = []
  for line in f:
    # 改行を除いてスペース区切りにする
    line = line.rstrip()
    l = line.split()
    feat = []
    # データを読み込んで28x28に縮小
    img = cv2.imread(os.getcwd() + l[0])
    #img = cv2.resize(img, dsize = (IMAGE_SIZE, IMAGE_SIZE))
    feat.append(imgproc.get_hsv_stats(img))
    feat.append(imgproc.get_rgb_stats(img))
    feat = sum(feat, [])  #1次元の配列に変換する
    train_features.append(feat)
    train_image.append(img.astype(np.float32)/255.0)
    train_label.append(int(l[1]))
  f.close()


  #SVMによる学習
  # クラスオブジェクトを生成(Classification)
  model = sksvm.SVC(C=10, gamma=0.01)
  # 学習する
  model.fit(train_features, train_label)

  # 解析用に特徴量の詳細データを取得
  #for 
  #PCAで次元圧縮
  pca = PCA(n_components=2)
  pca.fit(train_features)
  pca_X = pca.transform(train_features)

  fig0, (ax0, ax1) = plt.subplots(1,2,figsize=(12,6))
  ax0.set_title("PCA with Correct ClassLabel")
  ax0.scatter(pca_X[:,0], pca_X[:,1], c=train_label)

  Z = model.predict(train_features)
  ax1.set_title("PCA with SVN Result ClassLabel")
  ax1.scatter(pca_X[:,0], pca_X[:,1], c=Z)

  plt.savefig('./log/svm/pca.png')
  plt.figure()

  #テストデータの取得
  f = open(test_dir, 'r')
  test_label = []
  test_features = []
  test_path = []
  for line in f:
    line = line.rstrip()
    l = line.split()
    feat = []
    img = cv2.imread(os.getcwd() + l[0])
    #img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    feat.append(imgproc.get_hsv_stats(img))
    feat.append(imgproc.get_rgb_stats(img))
    feat = sum(feat, [])  #1次元の配列に変換する
    test_features.append(feat)
    test_label.append(int(l[1]))
    test_path.append(os.getcwd() + l[0])
  f.close()

  # 予測する
  predict_y = model.predict(test_features)
  print(predict_y)

  sum_accuracy = 0.0
  for i in range(np.asarray(test_label).shape[0]):
    print("label:", test_label[i], "result:", predict_y[i])
    if test_label[i] == predict_y[i]:
        sum_accuracy += 1
  sum_accuracy /= np.asarray(test_label).shape[0]
  print("accuracy: ", sum_accuracy)

  #結果をhtmlファイル出力
  result_dict = {'acc':0, 'n_img':0, 'n_feat':0 }
  result_dict['acc'] = sum_accuracy
  result_dict['n_img'] = np.asarray(train_image).shape[0]
  result_dict['n_feat'] = np.asarray(test_features).shape[1]
  export.svm_html(result_dict, test_label, np.asarray(test_path), predict_y)

