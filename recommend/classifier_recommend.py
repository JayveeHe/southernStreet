# -*- coding: utf-8 -*-

"""使用svm推荐"""

import os
import sys
import scipy

# project path
project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
data_path = '%s/data' % (project_path)

# project import
sys.path.append(project_path)
from log.get_logger import logger, Timer
from user_modeling.classifier_selection import generate_X_y_arrays
from recommend.intersection import intersect


@Timer
def train_svm(clf,
              f_train_set='%s/train_combined_vec_data.csv' % (data_path)):
    """
    训练SVM分类器

    Args:
        clf: 分类器
        f_train_set: string, 训练集文件
    Returns:
        clf: 分类器
    """
    from sklearn import cross_validation
    (X, y) = generate_X_y_arrays(f_train_set)
    # 简单验证
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    logger.info('SVM classifier simple cross-validated scores ars %s' % (scores))

    # 训练
    clf.fit(X, y)
    logger.info('SVM classifier(%s) fit Done. Best params are %s with a best score of %0.2f' % (clf, clf.best_params_, clf.best_score_))

    return clf


@Timer
def train_LR(clf,
              f_train_set='%s/train_combined_vec_data.csv' % (data_path)):
    """
    训练LR分类器

    Args:
        clf: 分类器
        f_train_set: string, 训练集文件
    Returns:
        clf: 分类器
    """
    from sklearn import cross_validation
    (X, y) = generate_X_y_arrays(f_train_set)

    # 简单验证
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    logger.info('LR classifier simple cross-validated scores ars %s' % (scores))

    # 训练
    clf.fit(X, y)
    logger.info('LR classifier(%s) fit Done. And Coef are: %s' % (clf, clf.coef_)) 

    return clf


@Timer
def train_clf(clf,
              f_train_set='%s/train_combined_vec_data.csv' % (data_path)):
    """
    训练分类器

    Args:
        clf: 分类器
        f_train_set: string, 训练集文件
    Returns:
        clf: 分类器
    """
    from sklearn import cross_validation
    (X, y) = generate_X_y_arrays(f_train_set)

    # 简单验证
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    logger.info('Classifier simple cross-validated scores ars %s' % (scores))

    # 训练
    clf.fit(X, y)
    logger.info('Classifier(%s) fit Done.' % (clf)) 

    return clf

@Timer
def generate_predict_result(clf,
                            f_predict='%s/predict_set/predict_result.csv' % (data_path),
                            f_vec_set='%s/predict_set/predict_combined_vec_data.csv' % (data_path),
                            f_uid_iid_set='%s/predict_set/predict_set.csv' % (data_path)):
    """
    生成预测结果

    Args:
        clf: 分类器
        f_predict: string, 存放预测结果
        f_vec_set: string, 存放待预测向量的文件名
        f_uid_iid_set: string, 存放与向量对应的user_id, item_id
    Returns:
        clf: 分类器
    """
    predict_X, predict_y = generate_X_y_arrays(f_vec_set)
    logger.debug('predict start.')
    predict_y = clf.predict(predict_X)
    logger.debug('predict done, predict result size=%s' % (len(predict_y)))

    with open(f_uid_iid_set, 'r') as fin, open(f_predict, 'w') as fout:
        counter = 0
        fin.readline()    # 忽略首行
        fout.write('user_id,item_id,tag')
        
        logger.debug('start store predict result')
        for line in fin:
            line_result = line.strip() + ',%s\n' % (predict_y[counter])
            fout.write(line_result)
            counter += 1

    if counter != len(predict_y):
        logger.error('predict result size:%s, but uid_iid_set size:%s' % (len(predict_y), counter))
    else:
        logger.info('predict success, generate predict result in %s' % (f_predict))


@Timer
def generate_recommend_result(clf, f_predict_set, f_recommend_set):
    """
    根据预测结果生成推荐结果

    Args:
        clf: 分类器
        f_predict_set: string, 存放预测结果
        f_recommend_set: string, 存放推荐结果
    Returns:
        clf: 分类器
    """
    with open(f_predict_set, 'r') as fin, open(f_recommend_set, 'w') as fout:
        fin.readline()    # 忽略首行
        fout.write('user_id,item_id\n')

        counter = 0
        for line in fin:
            cols = line.strip().split(',')
            if cols[-1] == '1':
                counter += 1
                fout.write('%s,%s\n' % (cols[0], cols[1]))

    logger.info('generate recommend result Done. Total: %s' % (counter))


if __name__ == '__main__':
    store_path = '%s/classifier_data_1218' % (data_path)
    f_train_set = '%s/train_set.csv' % (store_path)

    # Randomized Parameter Optimization
    from sklearn import svm, grid_search
    randomized_parameter = {'kernel':['rbf'], 'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1)}
    clf = grid_search.RandomizedSearchCV(svm.SVC(), randomized_parameter)
    f_predict_set = '%s/predict_result_SVM.csv' % (store_path)
    f_recommend_set = '%s/recommend_result_SVM.csv' % (store_path)
    clf = train_svm(clf, f_train_set)
    clf = generate_predict_result(clf, f_predict_set)
    generate_recommend_result(clf, f_predict_set, f_recommend_set)
    intersect(f_recommend_set)

    # L2范式线性回归
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1000, penalty='l2', tol=0.01)
    f_predict_set = '%s/predict_result_LR.csv' % (store_path)
    f_recommend_set = '%s/recommend_result_LR.csv' % (store_path)
    clf = train_LR(clf, f_train_set)
    clf = generate_predict_result(clf, f_predict_set)
    generate_recommend_result(clf, f_predict_set, f_recommend_set)
    intersect(f_recommend_set)
    
    # 随机森林
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    f_predict_set = '%s/predict_result_RandomForest.csv' % (store_path)
    f_recommend_set = '%s/recommend_result_RandomForest.csv' % (store_path)
    clf = train_clf(clf, f_train_set)
    clf = generate_predict_result(clf, f_predict_set)
    generate_recommend_result(clf, f_predict_set, f_recommend_set)
    intersect(f_recommend_set)
