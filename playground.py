import imagenet

if __name__ == '__main__':
    p = '/Users/ethan/datasets/ImageNet_tiny'
    dataset = imagenet.ImageNet(p)
    dataset.create_tfrecords(p, 192, 1000)