# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy as np

FLAGS = None


def main(_):
  # Import data
  #train-x = np.fromfile('./data/train-x.dat',float32)
  #train-y = np.fromfile('./data/train-y.dat',float32)
  test_x = np.load('./data/test-x.npy')
  test_y = np.load('./data/test-y.npy')
  print('x:', test_x.shape)
  print('y:', test_y.shape)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
   # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  saver = tf.train.Saver()
  saver.restore(sess, "/tmp/model.ckpt")
  
  print("W:", W.shape, W.eval())
  print("b:", b.shape, b.eval())
  np.save("./data/W", W.eval())
  np.save("./data/b", b.eval())
  print(sess.run(accuracy, feed_dict={x: test_x,y_: test_y}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)