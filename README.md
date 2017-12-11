ubuntu 16.04 LTS  内存32GB
处理器 Intel® Core™ i5-6600K CPU @ 3.50GHz × 4 
一块显卡 GeForce GTX TITAN X

python版本2.7
## Requirements
- cv2
- numpy
- pandas
- imageio
- skimage
- tqdm 
- keras
- tensorflow

######################################
#        ---features.py              #
#        |                           #
#        ---main.py                  #
#        |                           #
#文件结构----video2img.py              #
#        |                           #
#        ---train文件夹               #
#        |                           #
#        ---test_B文件夹              #
######################################

需更改video2img.py 第12行所要处理视频的绝对路径

## 处理视频，耗时约6分钟
  运行python video2img.py
## 获取训练集features ，耗时约5小时
  运行python features.py
## 训练并输出预测，耗时约5分钟
  运行python main.py
