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

    dataset.create_tfrecords(os.path.join(p, 'tfrecords'), 192, 10)

