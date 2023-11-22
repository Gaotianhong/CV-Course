import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models import svm
from utils import load_cifar10_data, extract_HOG_features


if __name__ == '__main__':

    # 数据集路径
    data_dir = 'data/cifar-10-batches-py'
    isHOG = True

    start_time = time.time()
    date_time = time.localtime(start_time)
    print(time.strftime("%Y-%m-%d %H:%M:%S", date_time))

    X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)
    print('train_images:{}, y_train:{}, test_images:{}, y_test:{}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    # 提取 HOG 特征
    if isHOG:
        X_train = extract_HOG_features(X_train)
        X_test = extract_HOG_features(X_test)
        print('train_hog_features:{}, test_hog_features:{}'.format(X_train.shape, X_test.shape))

    # 数据预处理和标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    print('X_train:{}, X_test:{}'.format(X_train.shape, X_test.shape))

    # 创建一个SVM分类器
    mysvm = svm.LinearSVM()

    # 在训练集上训练SVM分类器
    if isHOG:
        loss_hist = mysvm.train(X_train, y_train, learning_rate=1e-2, reg=1e-3, num_iters=2500)  # SVM + HOG
        plt.figure(figsize=(6, 6))
        plt.plot(loss_hist)
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        plt.savefig('results/svm_hog.png', dpi=500)
    else:
        loss_hist = mysvm.train(X_train, y_train, learning_rate=1e-5, reg=1e-3, num_iters=2000)  # SVM
        plt.figure(figsize=(6, 6))
        plt.plot(loss_hist)
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        plt.savefig('results/svm.png', dpi=500)

    # 在测试集上进行预测
    test_predictions = mysvm.predict(X_test)

    # 计算分类准确度
    print('\naccuracy score:{}'.format(accuracy_score(y_test, test_predictions)))

    end_time = time.time()
    print('Run time:{}s'.format(end_time - start_time))
