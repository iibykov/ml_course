import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from matplotlib import pyplot as plt


# Отрисовать ROC кривую
def calc_and_plot_roc(y_true, y_pred_proba):
    # Посчитать значения ROC кривой и значение площади под кривой AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.title('Receiver Operating Characteristic', fontsize=15)
    plt.xlabel('False positive rate (FPR)', fontsize=15)
    plt.ylabel('True positive rate (TPR)', fontsize=15)
    plt.legend(fontsize=15)


def prepare_adult_data():
    adult = pd.read_csv('./adult.data',
                        names=['age', 'workclass', 'fnlwgt', 'education',
                               'education-num', 'marital-status', 'occupation',
                               'relationship', 'race', 'sex', 'capital-gain',
                               'capital-loss', 'hours-per-week', 'native-country', 'salary'])

    # Избавиться от лишних признаков
    adult.drop(['native-country'], axis=1, inplace=True)
    # Сконвертировать целевой столбец в бинарные значения
    adult['salary'] = (adult['salary'] != ' <=50K').astype('int32')
    # Сделать one-hot encoding для некоторых признаков
    adult = pd.get_dummies(adult,
                           columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                    'sex'])

    # Нормализовать нуждающиеся в этом признаки
    a_features = adult[['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']].values
    norm_features = (a_features - a_features.mean(axis=0)) / a_features.std(axis=0)
    adult.loc[:, ['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']] = norm_features

    # Разбить таблицу данных на матрицы X и y
    X = adult[list(set(adult.columns) - set(['salary']))].values
    y = adult['salary'].values

    # Добавить фиктивный столбец единиц (bias линейной модели)
    X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

    return X, y


# 3.9.1
X, y = prepare_adult_data()
clf = LogisticRegression(random_state=0).fit(X, y)
y_pred = clf.predict(X)
print('f1_score: ', f1_score(y, y_pred))

# 3.9.2
print('confusion_matrix:\n', confusion_matrix(y, y_pred))

# 3.9.3
y_pred_proba = clf.predict_proba(X)[:, 1]
# calc_and_plot_roc(y, y_pred_proba)
print('roc_auc_score: ', roc_auc_score(y, y_pred_proba))

# 3.9.4
clf_without_reg = LogisticRegression(penalty='none', solver='lbfgs').fit(X, y)
y_pred = clf_without_reg.predict(X)
print('f1_score: ', f1_score(y, y_pred))

# 3.9.5
# f1_score_max = 0.0
# for strength in np.arange(0.01, 1, 0.01):
#     clf_c = LogisticRegression(random_state=0, penalty='l2', C=strength, solver='lbfgs').fit(X, y)
#     y_pred_c = clf.predict(X)
#     f1_score_cur = f1_score(y, y_pred_c)
#     if f1_score_cur > f1_score_max :
#         f1_score_max = f1_score_cur
#
# print('f1_score_max: ', f1_score_max)


# 3.9.6
def prepare_adult_data_3_9_6():
    adult = pd.read_csv('./adult.data',
                        names=['age', 'workclass', 'fnlwgt', 'education',
                               'education-num', 'marital-status', 'occupation',
                               'relationship', 'race', 'sex', 'capital-gain',
                               'capital-loss', 'hours-per-week', 'native-country', 'salary'])

    # заменяем на other
    g = adult.groupby('native-country')['native-country']
    gg = g.transform('size')
    adult.loc[gg < 100, 'native-country'] = 'Other'

    # Сконвертировать целевой столбец в бинарные значения
    adult['salary'] = (adult['salary'] != ' <=50K').astype('int32')

    # Сделать one-hot encoding для некоторых признаков
    adult = pd.get_dummies(adult,
                           columns=['native-country', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                    'sex'])

    # Нормализовать нуждающиеся в этом признаки
    a_features = adult[['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']].values
    norm_features = (a_features - a_features.mean(axis=0)) / a_features.std(axis=0)
    adult.loc[:, ['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']] = norm_features

    # Разбить таблицу данных на матрицы X и y
    X = adult[list(set(adult.columns) - set(['salary']))].values
    y = adult['salary'].values

    # Добавить фиктивный столбец единиц (bias линейной модели)
    X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

    return X, y


X, y = prepare_adult_data_3_9_6()
clf = LogisticRegression(random_state=0).fit(X, y)
y_pred = clf.predict(X)
print('3.9.6 f1_score: ', f1_score(y, y_pred))

# 3.9.7
X, y = prepare_adult_data()
clf = LogisticRegression(random_state=42).fit(X, y)
y_pred = clf.predict(X)
print('3.9.7 f1_score: ', f1_score(y, y_pred))
