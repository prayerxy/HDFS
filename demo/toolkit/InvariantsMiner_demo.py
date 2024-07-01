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

from InvariantsMiner import InvariantsMiner
import dataloader, preprocessing

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file
epsilon = 0.5 # threshold for estimating invariant space

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='sequential')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)
    x_test = feature_extractor.transform(x_test)

    model = InvariantsMiner(epsilon=epsilon)
    model.fit(x_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)
    
    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)

