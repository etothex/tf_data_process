# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Gyq
# python 3.5

import tensorflow as tf
import os

dir = os.path.dirname(__file__)
# create a reader
reader = tf.TFRecordReader()
for d in ['train', 'validation', 'test']:
    # create a queue
    filename_queue = tf.train.string_input_producer([os.path.join(dir, 'mnist_data/mnist_'+d+'.tfrecords')])

    # 读取一个样例。也可以使用read_up_to函数一次读取多个样例
    _, serialized_example = reader.read(filename_queue)
    # 解析一个样例。解析多个可用parse_example函数
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)})
    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    pixels = tf.cast(features['pixels'], tf.int32)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(3):
        print(sess.run([image, label, pixels]))
