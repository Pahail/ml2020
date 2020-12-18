import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error


def plot_correlations_matrix(dataframe):
    """Строит матрицу корреляций признаков"""
    correlation_matrix = dataframe.corr().round(2)
    sns.set(rc={'figure.figsize': (14, 14)})
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.grid()
    plt.show()


def get_most_important_features(dataframe, target_feature, max_features=4):
    """Метод получает список из самых значисмых признаков"""
    correlation_matrix = dataframe.corr()
    sorted_df = correlation_matrix[target_feature].sort_values(ascending=False)
    del sorted_df[target_feature]
    # print(list(sorted_df.index.values))
    return np.array(sorted_df.index.values)[0:max_features]


def plot_gist(dataframe, target):
    """Построение гистограм"""
    sns.set(rc={'figure.figsize': (10, 8)})
    sns.distplot(dataframe[target], bins=30)
    plt.grid()
    plt.show()


def draw_dependency_ratio(dataframe, list_of_features, target_feature):
    """Отрисовываем зависимость от важных признаков"""
    figure = plt.figure(figsize=(16, 8))
    for i, col in enumerate(list_of_features):
        ax = figure.add_subplot(1, 4, i + 1)
        x = dataframe[col]
        y = dataframe[target_feature]
        ax.scatter(x, y, marker='o')
        ax.title.set_text(col)
        ax.set_xlabel(col)
        ax.set_ylabel('Chance of Admit')
    plt.show()


def print_model_stat(model, train_x, train_y, test_x, test_y, show_error=True):
    y_train_predict = model.predict(train_x)
    rmse = (np.sqrt(mean_squared_error(train_y, y_train_predict)))
    print("The model performance for training set")
    print('RMSE is {}, accuracy is {}'.format(rmse, model.score(train_x, train_y)))

    y_test_predict = model.predict(test_x)
    rmse = (np.sqrt(mean_squared_error(test_y, y_test_predict)))
    print("The model performance for training set")
    print('RMSE is {}, accuracy is {}'.format(rmse, model.score(test_x, test_y)))

    errors = np.array(test_y - y_test_predict)
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns.distplot(errors, bins=100)
    if show_error:
        plt.show()


if __name__ == '__main__':
    print("Домашнее задание по лекции 2")

    df = pd.read_csv('./Admission_Predict_Ver1.1.csv')
    df = df.rename(columns={"Chance of Admit ": "Chance of Admit"})
    # print(df.head())
    if np.array(df.isnull().sum()).sum() > 0:
        raise ValueError("Data contains None value")

    # plot_correlations_matrix(df)
    important_features = get_most_important_features(df, "Chance of Admit")
    # plot_gist(df, "Chance of Admit")

    draw_dependency_ratio(dataframe=df,
                          list_of_features=important_features,
                          target_feature="Chance of Admit")

    x_full = pd.DataFrame(np.c_[df[important_features]], columns=important_features)
    y_full = df['Chance of Admit']
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2,random_state=5)

    print("The Linear Regression")
    lin_model = LinearRegression()
    lin_model.fit(x_train, y_train)
    print_model_stat(lin_model, x_train, y_train, x_test, y_test, False)
    print("\n")

    print("The Linear Ridge Regression")
    lin_model_ridge = Ridge()
    lin_model_ridge.fit(x_train, y_train)
    print_model_stat(lin_model_ridge, x_train, y_train, x_test, y_test, False)
    print("\n")

    print("The Linear lasso Regression")
    lin_model_lasso = Lasso()
    lin_model_lasso.fit(x_train, y_train)
    print_model_stat(lin_model_lasso, x_train, y_train, x_test, y_test, False)
    print("\n")

