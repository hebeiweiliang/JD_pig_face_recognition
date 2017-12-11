#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
import random
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
#################################################################################
## 加载特征
n_class = 30
features = np.load('features.npy')
y = np.load('y.npy')

########################################################################################
## 训练
inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(features, y, batch_size=64, epochs=2, validation_split=0.1)
##########################################################################################
## 预测集加载
width = 299
n_test = 3000 #测试集大小
X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)
k = 0
l = []
for infile in glob.glob('test_B/*.JPG'):
    im = cv2.imread(infile)
    X_test[k] = cv2.resize(im, (width, width))
    f, ext = os.path.splitext(infile)
    l.append(f[7:])
    k = k+1

## 采用预训练好模型获取特征
def get_features(MODEL, data=X_test):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=32, verbose=1)
    return features
print('获取训练图片卷积后的特征...')

inception_features_test = get_features(InceptionV3, X_test)
xception_features_test = get_features(Xception, X_test)
features_test = np.concatenate([inception_features_test, xception_features_test], axis=-1)
y_pred = model.predict(features_test, batch_size=64)
############################################################################################
## 预测结果写入result.csv
o=0
for j in range(3000):
    datas = []
    for i in range(1,31):
        datas.append([l[o],i,'%.9f' %y_pred[o][i-1]],)
    f = open('result.csv','a')
    writer = csv.writer(f)
    writer.writerows(datas)
    o=o+1
    f.close()
print('完成')
