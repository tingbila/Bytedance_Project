# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


# models/FmLayer.py
import tensorflow as tf
from tensorflow.keras import layers

# FM二阶交叉项的计算流程
class FMInteractionLayer(layers.Layer):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.V = self.add_weight(
            shape=(input_shape[-1], self.embedding_dim),
            initializer="random_normal",
            trainable=True
        )
        print(self.V)
        print(self.V.shape)  # (234, 5)


    def call(self, x):
        # a*b = [(a+b)^2 - (a^2+b^2)]/2
        xv = tf.matmul(x, self.V)
        xv_square = tf.square(xv)
        x_square_v_square = tf.matmul(tf.square(x), tf.square(self.V))
        fm_second_order = 0.5 * tf.reduce_sum(xv_square - x_square_v_square, axis=1, keepdims=True)
        return fm_second_order



if __name__ == '__main__':
    # 假设输入维度为 6，V矩阵的embedding_dim 为5
    embedding_dim = 5
    input_dim = 6
    batch_size = 2

    # 创建测试输入 x：batch_size x input_dim
    x_input = tf.constant([[1., 0., 1., 0., 2., 0.],
                           [0., 1., 0., 3., 0., 1.]], dtype=tf.float32)

    # 初始化层
    fm_layer = FMInteractionLayer(embedding_dim=embedding_dim)
    _ = fm_layer.build(x_input.shape)

    # 调用 forward
    output = fm_layer(x_input)

    # 打印结果
    print("FM 二阶交叉输出 (batch_size x 1):")
    print(output.numpy())



# <Variable path=fm_interaction_layer/variable, shape=(6, 5), dtype=float32, value=[[ 0.02906959 -0.0113054   0.04794595 -0.00654231 -0.01800302]
#  [-0.03543176  0.05997542  0.05696088 -0.01322689  0.00932838]
#  [ 0.07129439  0.00479821  0.09070511  0.00947785  0.0679295 ]
#  [ 0.02142422 -0.0391124  -0.05202466 -0.11712926 -0.02297361]
#  [ 0.01637161  0.04605945  0.07401312 -0.05986025  0.00382077]
#  [ 0.0435433  -0.06175534  0.05068144 -0.00593142  0.09385119]]>
# (6, 5)
# FM 二阶交叉输出 (batch_size x 1):
# [[ 0.02832312]
#  [-0.01785501]]

