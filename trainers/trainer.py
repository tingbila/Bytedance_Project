# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


# trainers/trainer.py
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import os

# 设置 Pandas 显示选项，防止省略
pd.set_option('display.max_columns', None)   # 显示所有列
pd.set_option('display.max_rows', None)      # 显示所有行
pd.set_option('display.width', None)         # 自动调整显示宽度
pd.set_option('display.max_colwidth', None)  # 显示每列最大内容宽度


import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个


def train_and_evaluate(model, train_dataset, valid_dataset, test_dataset, epochs=10):
    model.compile(
        optimizer='adam',
        loss=['binary_crossentropy', 'binary_crossentropy'],
        metrics=[['accuracy'], ['accuracy']]
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
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=output_model_file,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=50,
        callbacks=callbacks
    )

    print("\nTest Evaluation:")
    # 评估
    test_loss = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Loss & Accuracy: {test_loss}")

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
    plt.plot(history.epoch, history.history['finish_accuracy'], label='Train Acc - Finish')
    plt.plot(history.epoch, history.history['val_finish_accuracy'], label='Val Acc - Finish')
    plt.plot(history.epoch, history.history['like_accuracy'], label='Train Acc - Like')
    plt.plot(history.epoch, history.history['val_like_accuracy'], label='Val Acc - Like')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()