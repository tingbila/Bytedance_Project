# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


from config.data_config import *

# trainers/trainer.py
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf

# 设置 Pandas 显示选项，防止省略
pd.set_option('display.max_columns', None)   # 显示所有列
pd.set_option('display.max_rows', None)      # 显示所有行
pd.set_option('display.width', None)         # 自动调整显示宽度
pd.set_option('display.max_colwidth', None)  # 显示每列最大内容宽度


import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个


def train_and_evaluate(model, train_ds, valid_ds, feat_columns):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name='AUC'), tf.keras.metrics.BinaryAccuracy(name='ACC')]
    )

    # 输出目录结构
    base_dir = './outputs'
    log_dir = os.path.join(base_dir, 'logs')
    callbacks_dir = os.path.join(base_dir, 'callbacks')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(callbacks_dir, exist_ok=True)
    output_model_file = os.path.join(callbacks_dir, 'best_model.keras')

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_AUC', patience=5, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=output_model_file,
            monitor='val_AUC',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]


    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # 然后再打印
    print(pd.DataFrame(history.history))
    print(history.epoch)

    # 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['loss'], label='Train Loss')
    plt.plot(history.epoch, history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, history.history['AUC'], label='Train AUC - Finish')
    plt.plot(history.epoch, history.history['val_AUC'], label='Val Acc - Finish')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()