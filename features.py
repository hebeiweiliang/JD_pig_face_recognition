#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
from random import shuffle
import os
import pylab
import imageio
import skimage
import sys
import csv
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input

imageio.plugins.ffmpeg.download()
## 处理视频
n_class = 30#分类个数
m = 984*30#图片计数

rate = 7 #数据增强
n = m*rate #训练集图片个数
width = 299#训练图片大小
b = [i for i in range(n)] #训练图片索引
shuffle(b) #训练图片索引洗牌
## 加载全部训练数据到内存加快训练，约占用60GB内存
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)
j = 0
print('加载训练图片...')
for i in tqdm(range(30)):
    for infile in glob.glob('train/'+str(i+1)+'/*.jpg'):
        for k in range(3):
            X[b[j]] = cv2.resize(cv2.imread(infile)[:,186*k:186*k+720,:], (width, width))
            y[b[j]][i] = 1
            j = j + 1
            X[b[j]] = cv2.resize(cv2.imread(infile)[:,100*k:100*k+980,:], (width, width))
            y[b[j]][i] = 1
            j = j + 1
        X[b[j]] = cv2.resize(cv2.imread(infile), (width, width))
        y[b[j]][i] = 1
        j = j + 1

# 采用预训练好模型获取特征
def get_features(MODEL, data=X):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=32, verbose=1)
    return features
## 获取训练集特征
print('获取训练图片特征...')
inception_features = get_features(InceptionV3, X)
xception_features = get_features(Xception, X)

features = np.concatenate([inception_features, xception_features], axis=-1)
np.save('features.npy',features)
np.save('y',y)
