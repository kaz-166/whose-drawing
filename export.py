#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_TRAINING_ACCURACY_GRAPH_PATH = './log/training_accuracy.png'
LOG_TRAINING_LOSS_GRAPH_PATH = './log/training_loss.png'
LOG_TRAINING_MODEL_PATH = './log/model.png'
LABEL_ANNOTATION_PATH = './label_annotation.txt'

#学習結果グラフの出力
def plot(history):
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

#結果をhtmlファイル出力
def html(result_dict, test_image, test_label, test_path, result, result_prob):
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
    f.write("学習データ画像枚数: " + str(result_dict['n_img']) + "枚<br>")
    f.write("最適化法: " + result_dict['opt'] + "<br>")
    f.write("活性化関数: " + result_dict['act_func'] + "<br>")

    f.write("<img src=" + os.getcwd() + LOG_TRAINING_ACCURACY_GRAPH_PATH + " alt=\"\"  height=\"400\"  />")
    f.write("<img src=" + os.getcwd() + LOG_TRAINING_LOSS_GRAPH_PATH + " alt=\"\"  height=\"400\"  />")
    f.write("<hr><h2>Neural Network Structure</h2><hr>")
    f.write("<img src=" + os.getcwd() + LOG_TRAINING_MODEL_PATH + " alt=\"\"  witdh=\"400\"  height=\"800\"  /> <br>")
    f.write("<hr><h2>Prediction Results</h2><hr>")
    f.write("<b>Prediction accuracy: " + str(int(result_dict['acc']*100)) + "[%]</b><br>")

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

