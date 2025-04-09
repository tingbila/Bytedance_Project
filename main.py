# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email :  mingyang.zhang@ushow.media

# 项目入口 main.py
# 用于训练和评估模型

from datasets.data_loader import load_dataset
from trainers.trainer import train_and_evaluate
import tensorflow as tf
from datasets.utils_tf import create_dataset
from config.data_config import *
from models.deepfm import DeepFM


if __name__ == "__main__":
    # 加载数据集
    data, train_ds, valid_ds, feat_columns = create_dataset(file=file, embed_dim=embed_dim)

    # 构建模型
    model = DeepFM(feat_columns,embed_dim)

    # 训练并评估
    train_and_evaluate(model, train_ds, valid_ds, feat_columns)
