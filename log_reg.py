from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt


def data_preprocessing():
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    cv = CountVectorizer(max_features=100)
    cv.fit(newsgroups_train['data'])
    x_train = cv.transform(newsgroups_train['data'])
    x_test = cv.transform(newsgroups_test['data'])

    y_train = newsgroups_train['target']
    y_test = newsgroups_test['target']
    # Оставляем только первые 2 класса
    train_mask = (y_train <= 1)
    test_mask = (y_test <= 1)

    x_train = x_train[train_mask]
    x_test = x_test[test_mask]

    y_train = y_train[train_mask]
    y_test = y_test[test_mask]
    return x_train, y_train, x_test, y_test


class MyLogisticRegression:

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.max_iter = 1000
        self.data_size = None
        self.w = None
        self.acc_train = np.zeros(self.max_iter)

    def fit(self, x_data, y_data):
        self.data_size = x_data.shape[0]
        self.w = np.random.rand(x_data.shape[1]) / 1000

        for i in range(self.max_iter):
            next = self.w - self.learning_rate * np.array(self.grad(self.w, x_data, y_data))
            self.acc_train[i] = self.accuracy(next, x_data, y_data)
            self.w = next
            assert not np.any(self.w == np.nan)

    def predict(self, test_data):
        return self.model(self.w, test_data)

    def calculate_accuracy(self, x_data, y_data):
        return self.accuracy(self.w, x_data, y_data)

    @staticmethod
    def sigma(w, x):
        return 1 / (1 + np.exp(-x.dot(w)))

    @staticmethod
    def model(w, X):
        return np.where(MyLogisticRegression.sigma(w, X) > 0.5, 1, -1)

    @staticmethod
    def accuracy(w, X, y):
        return sum(MyLogisticRegression.model(w, X) == y) / y.shape[0]

    @staticmethod
    def loss(w, X, y):
        N = w.shape[0]
        return 1 / N * np.sum(np.log(1 + np.exp(-y * X.dot(w))))

    @staticmethod
    def grad(w, X, y):
        N = w.shape[0]
        return np.array(-1 / N * np.sum(X.T.multiply(y).multiply(1 / (1 + np.exp(-y * X.dot(w)))).multiply(np.exp(-y * X.dot(w))), axis=1)).ravel()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data_preprocessing()
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    y_test_predicted = log_reg.predict(x_test)
    y_train_predicted = log_reg.predict(x_train)

    print("Библиотечня логистическая регрессия")
    print("Точность на обучающей выборке {}".format(sum(y_train_predicted == y_train) / len(y_train)))
    print("Точность на тестовой выборке {}".format(sum(y_test_predicted == y_test) / len(y_test)))

    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    my_log_reg = MyLogisticRegression()
    my_log_reg.fit(x_train, y_train)
    print("Моя логистическая регрессия")
    print("Точность на обучающей выборке {}".format(my_log_reg.calculate_accuracy(x_train, y_train)))
    print("Точность на тестовой выборке {}".format(my_log_reg.calculate_accuracy(x_test, y_test)))

