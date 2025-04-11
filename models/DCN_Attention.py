# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# AttentionCrossLayer：通过引入注意力机制动态地调整每一层Cross Layer的特征交叉
class AttentionCrossLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_layers, num_heads=1, key_dim=8):
        """
        初始化 AttentionCrossLayer。

        参数:
        - input_dim: 输入的特征维度
        - num_layers: Cross 层的数量
        - num_heads: MultiHeadAttention 的头数
        - key_dim: 每个头的维度
        """
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim

        # 定义 num_layers 个 MultiHeadAttention 层
        self.attention_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
            for _ in range(num_layers)
        ]

        # 为每一层Cross网络初始化权重（x * w）以及偏置（b）
        self.ws = [
            self.add_weight(
                shape=(input_dim, 1),  # 每层的权重矩阵大小为 (input_dim, 1)
                initializer='random_normal',  # 使用正态分布初始化
                trainable=True,
                name=f'cross_weight_{i}'  # 为每一层命名
            ) for i in range(num_layers)
        ]

        # 为每一层初始化偏置项
        self.bs = [
            self.add_weight(
                shape=(input_dim,),  # 每层的偏置大小为 (input_dim,)
                initializer='zeros',  # 初始化为零
                trainable=True,
                name=f'cross_bias_{i}'
            ) for i in range(num_layers)
        ]

    def call(self, x):
        """
        前向传播：每一层先通过注意力机制处理输入，然后执行交叉操作。

        参数:
        - x: 输入张量，形状为 [batch_size, input_dim]

        返回:
        - x: 经过多层Cross操作和Attention后得到的输出
        """
        x0 = x  # 保存原始输入，作为 Cross 操作的基础

        for i in range(self.num_layers):
            # 使用 Attention 机制：Q = 当前x，K/V = 原始输入x0
            # 注意：Query、Key、Value 都使用输入 x 和初始输入 x0
            attn_output = self.attention_layers[i](query=x[:, None, :], key=x0[:, None, :], value=x0[:, None, :])

            # 输出注意力计算结果
            # print('-------->attn_output------>')
            # print(attn_output)

            # 去掉序列长度维度，将输出转换为形状 [batch_size, feature_dim]
            attn_output = tf.squeeze(attn_output, axis=1)  # 去掉 seq_len 维度

            # 输出去掉序列长度后的注意力结果
            # print(attn_output)

            # 替代原始的 x0 * (x · w) + b + x
            # 在此，我们通过将注意力输出与权重矩阵相乘得到交叉后的特征
            xw = tf.matmul(attn_output, self.ws[i])  # 计算加权的交叉项，结果为 [batch_size, 1]

            # 替代原始的 Cross 公式：x = x0 * xw + b + attn_output
            # 动态地将每一层的 attention 输出与交叉项结合，生成新的特征表示
            x = x0 * xw + self.bs[i] + x

        return x  # 返回最终的输出



# 测试用例
if __name__ == '__main__':
    batch_size = 2
    input_dim = 4
    num_layers = 2
    num_heads = 2
    key_dim = 4

    # 随机生成输入数据
    np.random.seed(42)
    inputs = np.random.rand(batch_size, input_dim).astype(np.float32)

    print("输入数据形状:", inputs.shape)
    print("输入数据:\n", inputs)

    # 实例化并测试 AttentionCrossLayer
    attn_cross = AttentionCrossLayer(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, key_dim=key_dim)
    output = attn_cross(inputs)

    print("\nAttentionCrossLayer 输出形状:", output.shape)
    print("AttentionCrossLayer 输出:\n", output.numpy())



# 输入数据形状: (2, 4)
# 输入数据:
#  [[0.37454012 0.9507143  0.7319939  0.5986585 ]
#  [0.15601864 0.15599452 0.05808361 0.8661761 ]]
#
# AttentionCrossLayer 输出形状: (2, 4)
# AttentionCrossLayer 输出:
#  [[-0.4463965  -0.23177831 -0.38383034 -0.136402  ]
#  [ 0.06484402 -0.12796074 -0.1626702  -0.03800344]]