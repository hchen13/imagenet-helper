import json
import os
import random
from functools import partial
from glob import glob
import tensorflow as tf

import cv2
from tqdm import tqdm

from imagenet.tfannotations import serialize_example, read_tfrecord

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

    def _scan_image_files(self):
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'valid')
        self._train_images, self._valid_images = self._find_images()
        print(f"[ImageNet] {len(self._train_images)} training and "
              f"{len(self._valid_images)} validation image files detected.\n")

    def _scan_tfrecords(self, dir: str=None):
        if dir is None:
            dir = self.root_dir
        train_pattern = 'imagenet_*_train*.tfrecords'
        valid_pattern = 'imagenet_*_valid*.tfrecords'
        self._train_records = glob(os.path.join(dir, train_pattern))
        self._valid_records = glob(os.path.join(dir, valid_pattern))
        print(f"[ImageNet] {len(self._train_records)} training and "
              f"{len(self._valid_records)} validation tfrecords files detected.\n")

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
        print("[ImageNet] initializing ImageNet dataset...\n")
        metadata_dir = os.path.join(ROOT_DIR, 'metadata')
        self.valid_gt = json.load(open(os.path.join(metadata_dir, 'validation_ground_truths.json')))
        self.valid_blacklist = json.load(open(os.path.join(metadata_dir, 'validation_blacklist.json')))
        self.root_dir = dataset_dir
        self.lookup = Lookup()
        self._scan_image_files()
        self._scan_tfrecords()

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

        valid_gen = partial(self.valid_generator, image_size=image_size)
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

    def create_tfrecords(self, image_size: int, chunk_size: int=None, dir: str=None):
        if dir is None:
            dir = self.root_dir
        train_files = list(self._train_images)
        valid_files = list(self._valid_images)
        random.shuffle(train_files)
        random.shuffle(valid_files)
        if chunk_size:
            print(f"[ImageNet] creating tfrecords files in chunks so that each file will contain at most {chunk_size} images")
            chunk_id = 1
            for i in range(0, len(train_files), chunk_size):
                print(f"\ttrain set chunk #{chunk_id}.\n")
                filename = f'imagenet_{image_size}_train{chunk_id}.tfrecords'
                image_list = train_files[i : i + chunk_size]
                self._create_tfrecords(dir, filename, image_size, image_list)
                chunk_id += 1

            chunk_id = 1
            for i in range(0, len(valid_files), chunk_size):
                print(f"\tvalid set chunk #{chunk_id}.\n")
                filename = f'imagenet_{image_size}_valid{chunk_id}.tfrecords'
                image_list = valid_files[i : i + chunk_size]
                self._create_tfrecords(dir, filename, image_size, image_list)
                chunk_id += 1
            print("[ImageNet] creation complete.\n")
            return

        print("[ImageNet] creating tfrecords file for training set...")
        self._create_tfrecords(dir, f'imagenet_{image_size}_train.tfrecords', image_size, train_files)
        print("[ImageNet] creating tfrecords file for validation set...")
        self._create_tfrecords(dir, f'imagenet_{image_size}_valid.tfrecords', image_size, valid_files)
        print("[ImageNet] creation complete!\n")

    def _create_tfrecords(self, record_dir, filename, image_size, image_files):
        os.makedirs(record_dir, exist_ok=True)
        path = os.path.join(record_dir, filename)
        with tf.io.TFRecordWriter(path) as writer:
            for path in tqdm(image_files):
                image = cv2.imread(path)
                image = cv2.resize(image, (image_size, image_size))
                _, label_index, _ = self._parse_image_path(path)
                serialized = serialize_example(image, label_index)
                writer.write(serialized)

    def from_tfrecords(self, batch_size: int=None, path: str=None):
        if path is not None:
            self._scan_tfrecords(path)
        trainset, validset = None, None
        if len(self._train_records):
            raw = tf.data.TFRecordDataset(self._train_records)
            trainset = raw.map(read_tfrecord)

        if len(self._valid_records):
            raw = tf.data.TFRecordDataset(self._valid_records)
            validset = raw.map(read_tfrecord)

        if batch_size is not None:
            trainset = trainset.batch(batch_size)
            validset = validset.batch(batch_size)

        return trainset, validset

