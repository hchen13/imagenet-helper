import os

import imagenet

if __name__ == '__main__':
    p = '/path/to/your/ImageNet/'

    dataset = imagenet.ImageNet(dataset_dir=p)
    trainset, validset = dataset.from_tfrecords(batch_size=4, path=os.path.join(p, 'tfrecords'))