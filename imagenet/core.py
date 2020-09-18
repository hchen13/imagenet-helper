import json
import os
import random
from functools import partial
from glob import glob
import tensorflow as tf

import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Lookup:

    def __init__(self):
        lookup_path = os.path.join(ROOT_DIR, 'metadata/map_clsloc.txt')
        with open(lookup_path) as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        lookup = [line.split() for line in lines]
        lookup = [[a, int(b), c] for a, b, c in lookup]

        self.syn2id = {}
        self.id2syn = {}
        for i, (syn, id, text) in enumerate(lookup):
            self.syn2id[syn] = {
                'id': id,
                'text': text
            }
            self.id2syn[id] = {
                'syn': syn,
                'text': text
            }

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.syn2id[item]
        if isinstance(item, int):
            return self.id2syn.get(item)


class ImageNet:

    max_label_index = 1000
    min_label_index = 1

    def __init__(self, dataset_dir):
        """
        initialize the ImageNet dataset configurations by specifying the training and validation set directories.
        Notice that training set images with the same class are placed in a secondary folders named by its synset;
        while validation set images are placed in flat.
        e.g.:
        training: /path/to/train/n01774384/n01774384_11884.JPEG
        validation: /path/to/valid/ILSVRC2012_val_00019406.JPEG

        :param dataset_dir: path to the root directory of the dataset, should contain
        `metadata`, `train`, and `valid` folders
        """
        print("[ImageNet] initializing ImageNet dataset...")
        metadata_dir = os.path.join(ROOT_DIR, 'metadata')
        self.valid_gt = json.load(open(os.path.join(metadata_dir, 'validation_ground_truths.json')))
        self.valid_blacklist = json.load(open(os.path.join(metadata_dir, 'validation_blacklist.json')))
        self.train_dir = os.path.join(dataset_dir, 'train')
        self.valid_dir = os.path.join(dataset_dir, 'valid')
        self.root_dir = dataset_dir
        self.lookup = Lookup()
        self._train_images, self._valid_images = self._find_images()
        print(f"[ImageNet] {len(self._train_images)} training images and "
              f"{len(self._valid_images)} validation images detected.\n")

    @property
    def train_length(self):
        return len(self._train_images)

    @property
    def validation_length(self):
        return len(self._valid_images)

    def from_generator(self, image_size, batch_size, shuffle=False, take=None):
        """
        initialize dataset in the format of tf dataset in the generator fashion

        :param image_size: int, image size
        :param batch_size: int
        :param shuffle: boolean, whether to shuffle the list
        :param take: int if debugging else None, outputs `take` number of batches of data
        :return: trainset and validset
        """
        train_gen = partial(self.train_generator, image_size=image_size, shuffle=shuffle)
        trainset = tf.data.Dataset.from_generator(train_gen, (tf.uint8, tf.int32))
        trainset = trainset.batch(batch_size)
        trainset.shuffle(buffer_size=200)

        valid_gen = partial(self.valid_generator, image_size=image_size, shuffle=shuffle)
        validset = tf.data.Dataset.from_generator(valid_gen, (tf.uint8, tf.int32))
        validset = validset.batch(batch_size)
        if take:
            trainset = trainset.take(take)
            validset = validset.take(take)
        self.trainset = trainset
        self.validset = validset
        return trainset, validset

    def _gen(self, paths, image_size, shuffle):
        if shuffle:
            random.shuffle(paths)
        for i, path in enumerate(paths):
            image = cv2.imread(path)
            if image_size is not None:
                image = cv2.resize(image, (image_size, image_size))
            _, label_index, label_text = self._parse_image_path(path)
            yield image, label_index

    def train_generator(self, image_size=None, shuffle=False):
        paths = self._train_images
        return self._gen(paths, image_size, shuffle)

    def valid_generator(self, image_size=None, shuffle=False):
        paths = self._valid_images
        return self._gen(paths, image_size, shuffle)

    def _find_images(self):
        train_pattern = os.path.join(self.train_dir, '*', '*.JPEG')
        train_images = glob(train_pattern)
        valid_pattern = os.path.join(self.valid_dir, '*.JPEG')
        valid_images = glob(valid_pattern)
        return train_images, valid_images

    def _parse_image_path(self, path):
        """
        parse the image absolute path and retrieve image label information
        :param path: absolute path of either train/valid image
        :return: absolute path, label index, label text
        """
        relpath = os.path.relpath(path, self.root_dir)
        if relpath.startswith('train'):
            dirname = os.path.dirname(relpath)
            synset = os.path.basename(dirname)
            label = self.lookup[synset]
            label_id = label['id']
            text = label['text']
        elif relpath.startswith('valid'):
            filename = os.path.basename(relpath)
            label_id = self.valid_gt[filename]
            text = self.lookup[label_id]['text']
        else:
            raise ValueError(f"[ImageNet] Unusual path: {path}")
        return path, label_id, text
