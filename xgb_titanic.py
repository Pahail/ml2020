import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import xgboost
from sklearn.model_selection import GridSearchCV


def data_preprocessing(train_df, test_df):
    train_df.describe(include='O')
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    pd.crosstab(train_df['Title'], train_df['Sex'])

    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    pd.crosstab(train_df['Title'], train_df['Sex'])

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    guess_ages = np.zeros((2, 3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()

                age_mean = guess_df.mean()
                age_std = guess_df.std()
                age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass

    freq_port = train_df.Embarked.dropna().mode()[0]

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[
            (dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[
            (dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()
    print(train_df.head(10))
    print(X_train.shape, Y_train.shape, X_test.shape)
    return X_train, Y_train, X_test, Y_train


if __name__ == '__main__':
    train_df = pd.read_csv('./titanic/train.csv')
    test_df = pd.read_csv('./titanic/test.csv')
    combine = [train_df, test_df]

    X_train, Y_train, X_test, Y_test = data_preprocessing(train_df, test_df)

    model_xgb = xgboost.XGBClassifier()

    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'n_estimators': [500, 600, 700, 800],
    }

    grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=3, verbose=2)

    grid_search.fit(X_train, Y_train)

    acc_xgboost = grid_search.score(X_train, Y_train)
    xgb_pred = grid_search.predict(X_test)
    print(acc_xgboost)
    submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],
                               "Survived": xgb_pred})
    submission.to_csv('titanic_prediction_xgb.csv', index=False)
