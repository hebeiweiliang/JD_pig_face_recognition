import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
from random import shuffle
import os
import pylob
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
m = 0#图片计数
for video_index in range(n_class):
    #视频的绝对路径
    filename = 'path/to/train'
    os.mkdir(filename+'/'+str(video_index+1))
    vid = imageio.get_reader(filename +'/'+ '%d.mp4'%(video_index+1),  'ffmpeg')
    for num,im in enumerate(vid):
        if num %3 ==0:
            save_url = filename+'/'+str(video_index+1)+'/%d.jpg'%num           
            im = np.array(im)
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_url,im)
            m = m+1

rate = 7 #数据增强
n = m*rate #训练集图片个数

width = 299#训练图片大小
b = [i for i in range(n)] #训练图片索引
shuffle(b) #训练图片索引洗牌

## 加载全部训练数据到内存加快训练，约占用60GB内存
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)

j = 0
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

inception_features = get_features(InceptionV3, X)
xception_features = get_features(Xception, X)

features = np.concatenate([inception_features, xception_features], axis=-1)
## 获取特征后可释放内存
#X = 0

## train

inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(features, y, batch_size=64, epochs=2, validation_split=0.1)

## test
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

## 获取测试集features
inception_features_test = get_features(InceptionV3, X_test)
xception_features_test = get_features(Xception, X_test)

features_test = np.concatenate([inception_features_test, xception_features_test], axis=-1)

y_pred = model.predict(features_test, batch_size=64)

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
