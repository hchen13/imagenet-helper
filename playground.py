import imagenet

if __name__ == '__main__':
    if os.environ.get('AIBOX') is None:  # 本地开发
        p = '/Users/ethan/datasets/ImageNet_tiny/'
    else:                                # 训练机
        p = '/media/ethan/DataStorage/ImageNet/'

    p = '/Users/ethan/datasets/ImageNet_tiny'
    dataset = imagenet.ImageNet(p)
    dataset.create_tfrecords(p, 192, 1000)