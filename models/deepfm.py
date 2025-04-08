# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


# models/deepfm.py
from tensorflow.keras import layers, Model
from models.fm_layer import FMInteractionLayer

def build_deepfm_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))

    # 线性部分
    linear_part = layers.Dense(1)(inputs)

    # FM 二阶交互项
    fm_interactions = FMInteractionLayer(embedding_dim=5)(inputs)

    # Deep 部分
    x = layers.Dense(128)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    # 输出层
    deep_output = layers.Dense(1)(x)

    output1 = layers.Dense(1, activation='sigmoid', name='finish')(linear_part + fm_interactions + deep_output)
    output2 = layers.Dense(1, activation='sigmoid', name='like')(linear_part + fm_interactions + deep_output)

    model = Model(inputs=inputs, outputs=[output1, output2])
    return model