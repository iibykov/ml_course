import numpy as np
from numpy.linalg import inv

from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_curve, roc_auc_score

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
# from matplotlib import pyplot as plt

data = load_boston()
data['data'].shape
# print(data['DESCR'])


# ЗАДАЧА Реализовать функцию, осуществляющую матричные операции для получения theta
def linreg_linear(X, y):
    theta = inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    return theta


# Подготовить данные
X, y = data['data'], data['target']
X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

# Вычислить параметр theta
theta = linreg_linear(X, y)

# Сделать предсказания для тренировочной выборки
y_pred = X.dot(theta)


def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE = {mse:.2f}, RMSE = {rmse:.2f}')


# Посчитать значение ошибок MSE и RMSE для тренировочных данных
print_regression_metrics(y, y_pred)

print('---sklearn.linear_model---')
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)
print_regression_metrics(y, y_pred)

# 3.5.3 признак с наибольшим стандартным отклонением
print(data['feature_names'][np.argmax(X.std(axis=0)) + 1])
print(np.max(X.std(axis=0)))

# 3.5.4
X, y = data['data'], data['target']
theta = linreg_linear(X, y)
y_pred = X.dot(theta)
print_regression_metrics(y, y_pred)

# 3.5.5
B_index = 11
X = data.data[data.data[:, B_index] > 50]
y = data.target[data.data[:, B_index] > 50]
X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

theta = linreg_linear(X, y)
y_pred = X.dot(theta)
print_regression_metrics(y, y_pred)

# 3.5.6
X, y = data['data'], data['target']
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_norm = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

theta = linreg_linear(X_norm, y)
y_pred = X_norm.dot(theta)
print_regression_metrics(y, y_pred)
