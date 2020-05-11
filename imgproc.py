#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv2

#HSVそれぞれの平均と分散を取得する
def get_hsv_stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # hsv票色系に変換
    h,s,v = cv2.split(hsv) # 各成分に分割
    return [h.mean(), s.mean(), v.mean(), h.std(), s.std(), v.std()]

#RGBそれぞれの平均と分散を取得する
def get_rgb_stats(img):
    bgr = cv2.cvtColor(img, cv2.IMREAD_COLOR) # rgb票色系
    g, b, r = cv2.split(bgr)
    return [r.mean(), g.mean(), b.mean(), r.std(), g.std(), b.std()]
