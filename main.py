# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email :  mingyang.zhang@ushow.media

# 项目入口 main.py
# 用于训练和评估模型

from datasets.data_loader import load_dataset
from models.deepfm import build_deepfm_model
from trainers.trainer import train_and_evaluate
import tensorflow as tf



if __name__ == "__main__":
    # 加载数据集
    train_dataset, valid_dataset, test_dataset, input_dim = load_dataset(r"D:\\BaiduNetdiskDownload\\头条数据\\train_2.csv")

    # 构建模型
    model = build_deepfm_model(input_dim)
    tf.keras.utils.plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True,rankdir='BT')

    # 训练并评估
    train_and_evaluate(model, train_dataset, valid_dataset, test_dataset)
