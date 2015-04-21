# -*- coding: utf-8 -*-

"""使用svm推荐"""

import os
import sys
#import scipy
import time

# project path
project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
data_path = '%s/data' % (project_path)

# project import
sys.path.append(project_path)
from log.get_logger import logger, Timer
from user_modeling.classifier_selection import generate_X_y_arrays
from recommend.intersection import intersect


@Timer
def train(clf, f_train_set):
    """
    训练分类器

    Args:
        clf: 分类器
        f_train_set: fin, 训练集文件
    Returns:
        clf: 分类器
    """
    from sklearn import cross_validation
    (X, y) = generate_X_y_arrays(f_train_set)

    # 简单验证
    #logger.debug('Start simple cross-validate.')
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    logger.info('Classifier simple cross-validated(use train set) scores ars %s' % (scores))

    # 训练
    clf.fit(X, y)
    logger.info('Classifier(%s) fit Done.' % (clf)) 
    #logger.info('LR classifier(%s) fit Done. And Coef are: %s' % (clf, clf.coef_)) 
    #logger.info('SVM classifier(%s) fit Done. Best params are %s with a best score of %0.2f' % (clf, clf.best_params_, clf.best_score_))

    return clf


@Timer
def predict(clf, f_predict_vect, f_predict_id_set, f_predict_out):
    """
    根据预测数据，给出预测结果

    Args:
        clf: 分类器
        f_predict_vect: fin, 预测数
        f_predict_id_set: fin, 与预测数据对应的存放有user_id, item_id的文件
        f_predict_out: fout, 存放预测结果的文件
    Returns:
        f_predict_out: fout, 存放预测结果的文件
    """
    predict_X, predict_y = generate_X_y_arrays(f_predict_vect)
    logger.debug('predict start.')
    predict_y = clf.predict(predict_X)
    logger.debug('predict done, predict result size=%s' % (len(predict_y)))

    with open(f_predict_id_set, 'r') as fin, open(f_predict_out, 'w') as fout:
        counter = 0
        fin.readline()    # 忽略首行
        fout.write('user_id,item_id,tag')
        
        logger.debug('start store predict result')
        for line in fin:
            line_result = line.strip() + ',%s\n' % (predict_y[counter])
            fout.write(line_result)
            counter += 1

    if counter != len(predict_y):
        assert(counter == len(predict_y))
        logger.error('predict result size:%s, but uid_iid_set size:%s' % (len(predict_y), counter))
    else:
        logger.info('predict success, generate predict result in %s' % (f_predict_out))

    return f_predict_out


@Timer
def recommend(f_predict_out, f_recommend_set):
    """
    根据预测结果生成推荐结果

    Args:
        f_predict_set: string,fin, 存放预测结果
        f_recommend_set: string,fout, 存放推荐结果
    Returns:
        f_recommend_set_intersect: string,fout, 取交集后的推荐结果
    """
    with open(f_predict_out, 'r') as fin, open(f_recommend_set, 'w') as fout:
        fin.readline()    # 忽略首行
        fout.write('user_id,item_id\n')

        counter = 0
        for line in fin:
            cols = line.strip().split(',')
            if cols[-1] == '1':
                counter += 1
                fout.write('%s,%s\n' % (cols[0], cols[1]))
    logger.info('Generate recommend result Done. Total: %s' % (counter))

    f_recommend_set_intersect = intersect(f_recommend_set)  # 结果取交集
    return f_recommend_set_intersect


def test(f_recommend_intersect_set, f_real_buy_intersect_set):
    """
    测试推荐结果

    Args:
        f_recommend_intersect_set: fin, 取交集后的推荐结果
        f_real_buy_intersect_set: fin, 取交集后的真实购买结果
    Returns:
        scores: list, [f1_score, precision, recall]
    """
    prediction_set = set()
    reference_set = set()

    with open(f_real_buy_intersect_set, 'r') as fin:
        fin.readline()    # 忽略首行
        for line in fin:
            prediction_set.add(line.strip())
    with open(f_recommend_intersect_set, 'r') as fin:
        fin.readline()    # 忽略首行
        for line in fin:
            reference_set.add(line.strip())

    intersection_len = float(len(prediction_set.intersection(reference_set)))
    precision = intersection_len / len(reference_set)
    recall = intersection_len / len(prediction_set)
    f1_score = (2.0*precision*recall) / (precision+recall)
    logger.info('[test result] f1_score=%s, precision=%s, recall=%s' % (f1_score, precision, recall))

    return [f1_score, precision, recall]


if __name__ == '__main__':
    f_train_set = '%s/train_1219/combined_out.csv' % (data_path)
    """
    # for predict
    f_predict_vect = '%s/predict_1220/combined_vec_data.csv' % (data_path)
    f_predict_id_set = '%s/predict_1220/predict_set.csv' % (data_path)

    # L2范式线性回归
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1000, penalty='l2', tol=0.01)
    clf = train(clf, f_train_set)
    f_predict_out = predict(clf, f_predict_vect, f_predict_id_set, '%s/predict_1220/LR_predict_out.csv' % (data_path))
    recommend(f_predict_out, f_predict_out.replace('predict_out', 'recommend'))


    # Randomized Parameter Optimization
    from sklearn import svm, grid_search
    randomized_parameter = {'kernel':['rbf'], 'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1)}
    clf = grid_search.RandomizedSearchCV(svm.SVC(), randomized_parameter)
    clf = train(clf, f_train_set)
    f_predict_out = predict(clf, f_predict_vect, f_predict_id_set, '%s/predict_1220/SVM_predict_out.csv' % (data_path))
    recommend(f_predict_out, f_predict_out.replace('predict_out', 'recommend'))

    
    # 随机森林
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    clf = train(clf, f_train_set)
    f_predict_out = predict(clf, f_predict_vect, f_predict_id_set, '%s/predict_1220/RandomForest_predict_out.csv' % (data_path))
    recommend(f_predict_out, f_predict_out.replace('predict_out', 'recommend'))
    """

    """
    # for self test
    test_path = '%s/test_1206' % (data_path)
    f_predict_vect = '%s/test_combined.csv' % (test_path)
    f_predict_id_set = '%s/test_set.csv' % (test_path)
    f_real_buy_intersect_set = '%s/real_buy_intersect.csv' % (test_path)

    # L2范式线性回归
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1000, penalty='l2', tol=0.01)
    clf = train(clf, f_train_set)
    f_predict_out = predict(clf, f_predict_vect, f_predict_id_set, '%s/LR_predict_out.csv' % (test_path))
    f_recommend_set_intersect = recommend(f_predict_out, f_predict_out.replace('predict_out', 'recommend'))
    test(f_recommend_set_intersect, f_real_buy_intersect_set)


    # Randomized Parameter Optimization
    from sklearn import svm, grid_search
    randomized_parameter = {'kernel':['rbf'], 'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1)}
    clf = grid_search.RandomizedSearchCV(svm.SVC(), randomized_parameter)
    clf = train(clf, f_train_set)
    f_predict_out = predict(clf, f_predict_vect, f_predict_id_set, '%s/SVM_predict_out.csv' % (test_path))
    f_recommend_set_intersect = recommend(f_predict_out, f_predict_out.replace('predict_out', 'recommend'))
    test(f_recommend_set_intersect, f_real_buy_intersect_set)

    
    # 随机森林
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    clf = train(clf, f_train_set)
    f_predict_out = predict(clf, f_predict_vect, f_predict_id_set, '%s/RandomForest_predict_out.csv' % (test_path))
    f_recommend_set_intersect = recommend(f_predict_out, f_predict_out.replace('predict_out', 'recommend'))
    test(f_recommend_set_intersect, f_real_buy_intersect_set)
    """

    test_path = '%s/test_1206' % (data_path)
    f_real_buy_intersect_set = '%s/real_buy_intersect.csv' % (test_path)

    test('%s/RandomForest_recommend_intersect_categoryPopularity.csv'%(test_path), f_real_buy_intersect_set)

