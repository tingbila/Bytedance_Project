# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


# datasets/data_loader.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(file_path, batch_size=256):
    column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel",
                    "finish", "like", "music_id", "device", "time", "duration_time"]
    df = pd.read_csv(file_path, names=column_names)

    y = df[["finish", "like"]].values
    X = df.drop(columns=["finish", "like"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, (y_train[:, 0], y_train[:, 1])))
    train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, (y_valid[:, 0], y_valid[:, 1])))
    valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, (y_test[:, 0], y_test[:, 1])))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, valid_dataset, test_dataset, X.shape[1]