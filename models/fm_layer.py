# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


# models/fm_layer.py
import tensorflow as tf
from tensorflow.keras import layers

class FMInteractionLayer(layers.Layer):
    def __init__(self, embedding_dim):
        super(FMInteractionLayer, self).__init__()
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.V = self.add_weight(
            shape=(input_shape[-1], self.embedding_dim),
            initializer="random_normal",
            trainable=True
        )
        print(self.V.shape)  # (234, 5)

    def call(self, x):
        xv = tf.matmul(x, self.V)
        xv_square = tf.square(xv)
        x_square_v_square = tf.matmul(tf.square(x), tf.square(self.V))
        fm_second_order = 0.5 * tf.reduce_sum(xv_square - x_square_v_square, axis=1, keepdims=True)
        return fm_second_order