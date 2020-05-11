#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv2

#画像の彩度平均を算出する
def get_avg_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # hsv票色系に変換
    h,s,v = cv2.split(hsv) # 各成分に分割
    return s.mean()

#画像の輝度平均を算出する
def get_avg_value(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # hsv票色系に変換
    h,s,v = cv2.split(hsv) # 各成分に分割
    return v.mean()

#画像の色相平均を算出する
def get_avg_hue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # hsv票色系に変換
    h,s,v = cv2.split(hsv) # 各成分に分割
    return h.mean()