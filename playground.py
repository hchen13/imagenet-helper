import os
from glob import glob
import tensorflow as tf
from tqdm import tqdm

import imagenet
from imagenet.tfannotations import read_tfrecord

if __name__ == '__main__':
    if os.environ.get('AIBOX') is None:  # 本地开发
        p = '/Users/ethan/datasets/ImageNet_tiny/'
    else:                                # 训练机
        p = '/media/ethan/DataStorage/ImageNet/'

    dataset = imagenet.ImageNet(p)

    dataset.create_tfrecords(p, 192, 10)

    # trainset, validset = dataset.from_tfrecords(batch_size=4)
    # trainsize = 0
    # validsize = 0
    # for x, y in trainset:
    #     trainsize += x.shape[0]
    # for x, y in validset:
    #     validsize += x.shape[0]
    # print(trainsize, validsize)
