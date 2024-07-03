#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

from matplotlib import pyplot as plt
# 获取当前脚本所在目录的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取 Model 目录的路径
model_dir = os.path.join(current_dir, '../Model')

# 将 Model 目录添加到 sys.path
sys.path.append(model_dir)

# 现在可以导入 Model 目录中的模块
from PCA import PCA
import dataloader
import preprocessing

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'  # The structured log file
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file

if __name__ == '__main__':
    (x_train, _), (x_test, _),_ = dataloader.load_HDFS(struct_log,
                                                                label_file=None,
                                                                window='session',
                                                                train_ratio=0.5,
                                                                split_type='uniform')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf',
                                              normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)

    model = PCA(n_components=1)
    model.fit(x_train)
    y_test= model.predict(x_test)

    # print('Train validation:')
    # precision, recall, f1 = model.evaluate(x_train, y_train)

    # print('Test validation:')
    # precision, recall, f1 = model.evaluate(x_test, y_test)

    # # 绘制训练数据的 PCA 图
    # plot_pca(x_train, y_train, 'Train PCA Result')

    # 绘制测试数据的 PCA 图
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='coolwarm', marker='o', edgecolors='k')
    plt.title('PCA Result on Test Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')  # 如果你的 PCA 模型选择了两个主成分的话
    plt.colorbar(label='Anomaly Score')  # 可选的，用来显示异常分数的颜色条
    plt.grid(True)
    plt.show()

