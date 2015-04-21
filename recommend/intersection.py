# -*- coding: utf-8 -*-

"""对推荐结果和商品子集取交集"""

import MySQLdb
import arrow
import logging
import os
import sys
import json
from math import exp

# project path
project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
data_path = '%s/data' % (project_path)

# project import
sys.path.append(project_path)
from log.get_logger import logger


def intersect(f_result='%s/UserCF_recommend_3.csv' % (data_path),
              f_item_set='%s/tianchi_mobile_recommend_train_item.csv' % (data_path)):
    """
    对结果和给出的item_set取交集，剔除结果中不属于物品子集的

    Args:
        f_result: string, 原始的结果文件
                 -------------- content -------------
                | item_id,item_geohash,item_category |
                 ------------------------------------
        f_item_set: string, 阿里提供的物品集文件
                 ---- content ----
                | user_id,item_id |
                 -----------------
    Returns:
        fout_name: string, 取交集后的文件名
    """
    item_id_set = set()
    with open(f_item_set) as fin:
        fin.readline()  # 忽略首行
        for line in fin:
            cols = line.strip().split(',')
            item_id_set.add(cols[0])

    fout_name = f_result.replace('.csv', '_intersect.csv')
    counter = 0
    with open(f_result) as fin, open(fout_name, 'w') as fout:
        fout.write(fin.readline())  # 首行特殊处理
        for line in fin:
            cols = line.strip().split(',')
            if cols[1] in item_id_set:
                counter += 1
                fout.write(line)

    logger.info('intersect success, intersect size =%s and generate final result in %s' % (counter, fout_name))
    return fout_name


# def intersection_files(files_path=[]):
# if len(files_path) > 1:
#         for

def intersect_twofiles(fin1_path, fin2_path, fout_path):
    with open(fin1_path) as fin1, open(fin2_path) as fin2, open(fout_path, 'w') as fout:
        fin1.readline()  # 忽略首行
        tuple_list = []
        for line in fin1:
            cols = line.strip().split(',')
            meta_tuple = (cols[0], cols[1])
            tuple_list.append(meta_tuple)
        fout.write(fin2.readline())
        for line in fin2:
            cols = line.strip().split(',')
            meta_tuple = (cols[0], cols[1])
            if meta_tuple in tuple_list:
                fout.write('%s,%s\n' % (meta_tuple[0], meta_tuple[1]))
        logger.info('intersect %s and %s done, output=%s' % (fin1_path, fin2_path, fout_path))


if __name__ == '__main__':
    # intersect()
    # intersect('%s/UserCF_recommend_1.csv'%(data_path))
    # intersect('%s/test_set_1205-1206.csv' % (data_path))
    intersect_twofiles('%s/LR_recommend_intersect_ranked.csv' % (data_path),
                       '%s/RandomForest_recommend_intersect_ranked.csv' % (data_path),
                       '%s/intersect_temp.csv' % (data_path))
    intersect_twofiles('%s/intersect_temp.csv' % (data_path),
                       '%s/SVM_recommend_intersect_ranked.csv' % (data_path),
                       '%s/intersect_result.csv' % (data_path))