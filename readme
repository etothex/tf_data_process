### 简介
下载和读取mnist数据集，转化成tfrecord文件。
1. 读取mnist数据集-方法1
```python
from tensorflow.examples.tutorials.mnist import input_data
```
Maybe cause warning:  
> WARNING:tensorflow:From D:/python/tf_data_process/mnist2tfrecord.py:23: 
read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) 
is deprecated and will be removed in a future version.  
> Instructions for updating:  
> Please use alternatives such as official/mnist/dataset.py from tensorflow/models.  

In tensorflow\>=1.8, tensorflow.examples.tutorials is deprecated, you can get dataset.py from 
https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py or  
```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
```
2. 读取mnist数据集-方法2  
```python
import input_data
```
input_data.py是tensorflow提供的下载和读取MNIST数据集的源码。详情参考http://www.tensorfly.cn/tfdoc/tutorials/mnist_download.html