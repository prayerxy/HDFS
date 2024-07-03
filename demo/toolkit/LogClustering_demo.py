#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
# 获取当前脚本所在目录的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取 Model 目录的路径
model_dir = os.path.join(current_dir, '../Model')

# 将 Model 目录添加到 sys.path
sys.path.append(model_dir)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from LogClustering import LogClustering
import dataloader, preprocessing
struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'  # The structured log file
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file
max_dist = 0.3  # the threshold to stop the clustering process
anomaly_threshold = 0.3  # the threshold for anomaly detection
max_samples = 1000  # maximum number of samples to plot


def plot_clusters_3d(X, y_pred, title):
    # Randomly sample points if the dataset is larger than max_samples
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
        y_pred = y_pred[idx]

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=3, init='pca', learning_rate='auto', random_state=42)
    X_tsne = tsne.fit_transform(X)

    fig = plt.figure(figsize=(14, 6))

    ax = fig.add_subplot(121, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_pred, cmap='viridis', marker='o')
    ax.set_title(f'{title} - PCA')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    ax = fig.add_subplot(122, projection='3d')
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y_pred, cmap='viridis', marker='o')
    ax.set_title(f'{title} - t-SNE')
    ax.set_xlabel('Dim1')
    ax.set_ylabel('Dim2')
    ax.set_zlabel('Dim3')

    plt.show()


if __name__ == '__main__':
    (x_train, _), (x_test, _),_ = dataloader.load_HDFS(struct_log,
                                                                label_file=None,
                                                                window='session',
                                                                train_ratio=0.5,
                                                                split_type='uniform')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
    model.fit(x_train)  # Use only normal samples for training

    # print('Train validation:')
    # precision, recall, f1 = model.evaluate(x_train, y_train)

    # print('Test validation:')
    # precision, recall, f1 = model.evaluate(x_test, y_test)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    plot_clusters_3d(x_train, y_train_pred, 'Train Data')
    plot_clusters_3d(x_test, y_test_pred, 'Test Data')
