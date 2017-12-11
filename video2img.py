#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tqdm import tqdm
import os
import imageio

imageio.plugins.ffmpeg.download()
n_class = 30
print('处理视频...')
for video_index in tqdm(range(n_class)):
    #视频的绝对路径
    filename = '/home/new/桌面/pig2/train'
    os.mkdir(filename+'/'+str(video_index+1))
    vid = imageio.get_reader(filename +'/'+ '%d.mp4'%(video_index+1),  'ffmpeg')
    for num,im in enumerate(vid):
        if num %3 ==0:
            save_path = filename+'/'+str(video_index+1)+'/%d.jpg'%num           
            im = np.array(im)
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path,im)
            
