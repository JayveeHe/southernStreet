# -*- coding: utf-8 -*-

"""分类器选型"""

import os
import sys

# project path
project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
data_path = '%s/data' % (project_path)

# project import
sys.path.append(project_path)
from log.get_logger import logger, Timer


@Timer
def generate_X_y_arrays(f_train_set='%s/train_set.csv' % (data_path)):
    """
    生成分类器的训练集X 和标签集y

    Args:
        f_train_set: 训练集的csv文件
    Returns:
        X: training samples, size=[n_samples, n_features]
        y: class labels, size=[n_samples, 1]
    """
    from sklearn import preprocessing
    import numpy as np
    X = []
    y = []

    logger.debug('generate X, y arrays from %s ...' % (f_train_set))
    with open(f_train_set, 'r') as fin:
        fin.readline()  # 忽略首行
        for line in fin:
            cols = line.strip().split(',')
            X.append([float(i) for i in cols[1:]])
            y.append(int(cols[0]))  # tag在第一列，0 或 -1

    logger.debug('classifier input X_size=[%s, %s] y_size=[%s, 1]' % (len(X), len(X[0]), len(y)))
    X = preprocessing.scale(np.array(X))
    y = np.array(y)
    logger.debug('Scale params: mean=%s, std=%s' % (X.mean(axis=0), X.std(axis=0)))
    return X, y


@Timer
def tmp_generate_X_y_arrays(f_train_set='%s/train_set.csv' % (data_path)):
    """
    生成分类器的训练集X 和标签集y, 暂时删除其中某列

    Args:
        f_train_set: 训练集的csv文件
    Returns:
        X: training samples, size=[n_samples, n_features]
        y: class labels, size=[n_samples, 1]
    """
    from sklearn import preprocessing
    import numpy as np
    X = []
    y = []

    with open(f_train_set, 'r') as fin:
        fin.readline()  # 忽略首行
        for line in fin:
            cols = line.strip().split(',')
            X.append([float(i) for i in (cols[1:4]+cols[5:])])
            y.append(int(cols[0]))  # tag在第一列，0 或 -1

    logger.debug('classifier input X_size=[%s, %s] y_size=[%s, 1]' % (len(X), len(X[0]), len(y)))
    X = preprocessing.scale(np.array(X))
    y = np.array(y)
    return X, y


@Timer
def train_classifier(clf, X, y):
    """
    训练分类器

    Args:
        X: training samples, size=[n_samples, n_features]
        y: class labels, size=[n_samples, 1]
    Returns:
        clf: classifier, 训练完的分类器
    """
    from sklearn import grid_search, cross_validation
    import time

    """grid search 的结果
    clf.fit(X, y)
    #logger.info('Classifier fit Done. Best params are %s with a best score of %0.2f' % (clf.best_params_, clf.best_score_))
    #logger.info('And scores ars %s' % (clf.grid_scores_))
    """

    # 简单的交叉验证
    clf.fit(X, y)
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    logger.info('Classifier fit Done. And simple cross-validated scores ars %s' % (scores))

    # 十折法
    kf = cross_validation.KFold(len(X), n_folds=10)
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        logger.info('10 folds cross-validated scores is %s.' % (score))

    # 以 1/10的训练集作为新的训练集输入，并得出评分
    test_size = 0.9
    rs = cross_validation.ShuffleSplit(len(X), test_size=test_size, random_state=int(time.time()))
    for train_index, test_index in rs:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        logger.info('%s作为训练集输入， cross-validated scores is %s.' % (1-test_size, score))

    """
    # 以 1/100的训练集作为新的训练集输入，并得出评分
    test_size = 0.99
    rs = cross_validation.ShuffleSplit(len(X), test_size=test_size, random_state=int(time.time()))
    for train_index, test_index in rs:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        logger.info('%s作为训练集输入， cross-validated scores is %s.' % (1-test_size, score))
    """

    return clf


@Timer
def classifier_comparison(X, y):
    """
    分类器比较

    Args:
        X: training samples, size=[n_samples, n_features]
        y: class labels, size=[n_samples, 1]
    Returns:
        None
    """
    from sklearn import grid_search
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.lda import LDA
    from sklearn.qda import QDA
    from sklearn.linear_model import LogisticRegression
    import scipy

    # Exhaustive Grid Search
    exhaustive_parameters = {'kernel':['rbf'], 'C':[1, 10, 100, 1000], 'gamma':[1e-3, 1e-4]}
    clf_SVC_exhaustive = grid_search.GridSearchCV(SVC(), exhaustive_parameters)
    # Randomized Parameter Optimization
    randomized_parameter = {'kernel':['rbf'], 'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1)}
    clf_SVC_randomized = grid_search.RandomizedSearchCV(SVC(), randomized_parameter)

    names = ["Linear SVM", "RBF SVM",
             "RBF SVM with Grid Search", "RBF SVM with Random Grid Search", 
             "Decision Tree", "Random Forest", 
             "AdaBoost", "Naive Bayes", "LDA", "QDA"]
    classifiers = [
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        clf_SVC_exhaustive,
        clf_SVC_randomized,
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LDA(),
        QDA()]

    for name, clf in zip(names, classifiers):
        logger.info('Use %s:' % (name))
        train_classifier(clf, X, y)

    # 逻辑回归
    for C in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        logger.info('Use LR with l1 penalty, C=%s:' % (C))
        clf = LogisticRegression(C=C, penalty='l1', tol=0.01)
        clf = train_classifier(clf, X, y)
        logger.debug('coef matrix: %s' % (clf.coef_))

        logger.info('Use LR with l2 penalty, C=%s:' % (C))
        clf = LogisticRegression(C=C, penalty='l2', tol=0.01)
        clf = train_classifier(clf, X, y)
        logger.debug('coef matrix: %s' % (clf.coef_))


if __name__ == '__main__':
    (X, y) = generate_X_y_arrays('%s/train_1219/combined_out.csv' % (data_path))
    classifier_comparison(X, y)
    #(X, y) = tmp_generate_X_y_arrays('%s/train_combined_vec_data.csv' % (data_path))
    #classifier_comparison(X, y)

