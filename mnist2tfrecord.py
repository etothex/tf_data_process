# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Gyq
# python 3.5

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import input_data
import numpy as np
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


dir = os.path.dirname(__file__)
mnist_data = input_data.read_data_sets(os.path.join(dir, 'mnist_data'),
                                       dtype=tf.uint8,
                                       one_hot=True)

mnist_data = {'train': mnist_data.train, 'validation': mnist_data.validation, 'test': mnist_data.test}
for d in ['train', 'validation', 'test']:
    images = mnist_data[d].images
    labels = mnist_data[d].labels
    # images.shape is (55000, 784)
    pixels = images.shape[1]
    num_examples = mnist_data[d].num_examples

    # tfrecord文件的保存地址
    filename = os.path.join(dir, 'mnist_data/mnist_'+d+'.tfrecords')
    # create a writer to write tfrecord
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        # 转化成bytes(二进制数据)
        images_raw = images[index].tostring()
        # 将一个样例转化为Example Protocol Buffer
        example = tf.train.Example(features=tf.train.Features(feature={
            'pixels': _int64_feature(pixels),
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(images_raw)
        }))

        # 将Example写入tfrecord文件
        writer.write(example.SerializeToString())
    writer.close()
