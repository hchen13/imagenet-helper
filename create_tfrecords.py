import os

import imagenet

if __name__ == '__main__':
    p = '/path/to/your/ImageNet/'

    dataset = imagenet.ImageNet(dataset_dir=p)
    dataset.create_tfrecords(image_size=192, chunk_size=10, dir=os.path.join(p, 'tfrecords'))
