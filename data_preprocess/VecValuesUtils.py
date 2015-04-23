# coding=utf-8

import json
from datetime import datetime
import math
# import MySQLdb
import os
import sys

# project path


project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
data_path = '%s/data' % (project_path)

# project import
sys.path.append(project_path)
# from data_preprocess.MongoDB_Utils import MongodbUtils
from log.get_logger import logger, Timer


__author__ = 'Jayvee, jiaying.lu'

db_address = json.loads(open('%s/conf/DB_Address.conf' % (project_path), 'r').read())['MongoDB_Address']
MAX_BOUGHT_BEHAVIOR_COUNT = 120236


# 说明：从正样本中选取user_id与item_id，在建立向量的过程中只根据12-18日之前的数据，
# 即在数据库查询语句中添加时间戳《12-18-00的条件


data_dict = {'item_data': {}, 'user_data': {}}


def cal_item_popularity(mongo_train_user_collection, item_id, timerange=('2014-12-12', '2014-12-19')):
    """
    计算商品热门度，由于被除数都一样所以不再除以被购买商品总数，改为count的sigmoid形式

    :param item_id:
    :param stoptime:格式例如：'2014-12-18 00'
    :return:float类型的商品热度
    """

    if data_dict['item_data'].has_key(item_id):
        return data_dict['item_data'][item_id]['popularity']
    else:
        # mongodb = MongodbUtils(db_address, 27017)
        train_user = mongo_train_user_collection
        starttime = datetime.strptime(str(timerange[0]), '%Y-%m-%d')
        stoptime = datetime.strptime(str(timerange[1]), '%Y-%m-%d')
        bought_count = train_user.find(
            {'item_id': item_id, 'behavior_type': '4', "time": {'$gt': starttime, "$lt": stoptime}}).count()
        popularity = 1 / (1 + math.e ** (-bought_count)) - 0.5
        data_dict['item_data'][item_id] = {'popularity': popularity}
        return popularity


def cal_user_desire(mongo_train_user_collection, user_id, timerange=('2014-12-12', '2014-12-19')):
    """
    计算用户购买欲
    :param user_id:
    :return:float类型的用户购买欲
    """

    if data_dict['user_data'].has_key(user_id):
        max_count = data_dict['user_data'][user_id]['max_count']
        bought_count = data_dict['user_data'][user_id]['bought_count']
        if max_count == 0:
            return 0
        return float(bought_count) / float(max_count)
    else:
        train_user = mongo_train_user_collection
        starttime = datetime.strptime(str(timerange[0]), '%Y-%m-%d')
        stoptime = datetime.strptime(str(timerange[1]), '%Y-%m-%d')
        max_count = train_user.find({"user_id": user_id, "time": {"$lt": stoptime}}).count()
        bought_count = train_user.find(
            {"user_id": user_id, 'behavior_type': '4', "time": {'$gt': starttime, "$lt": stoptime}}).count()
        data_dict['user_data'][user_id] = {'max_count': max_count, 'bought_count': bought_count}
        if max_count == 0:
            return 0
        return float(bought_count) / float(max_count)


def cal_useritem_behavior_rate(mongo_train_user_collection, user_id, item_id, timerange=('2014-12-12', '2014-12-19')):
    """
    计算指定用户对指定商品的操作数占该用户总操作数的比重
    :param user_id:
    :param item_id:
    :return:
    """
    # logger.info('cal_useritem_behavior_rate: user_id = ' + user_id + '\titem_id = ' + item_id)

    # mongodb = MongodbUtils(db_address, 27017)
    train_user = mongo_train_user_collection
    starttime = datetime.strptime(str(timerange[0]), '%Y-%m-%d')
    stoptime = datetime.strptime(str(timerange[1]), '%Y-%m-%d')
    if data_dict['user_data'].has_key(user_id):
        max_count = data_dict['user_data'][user_id]['max_count']
    else:
        max_count = train_user.find({"user_id": user_id, "time": {"$lt": stoptime}}).count()
        bought_count = train_user.find(
            {"user_id": user_id, 'behavior_type': '4', "time": {'$gt': starttime, "$lt": stoptime}}).count()
        data_dict['user_data'][user_id] = {'max_count': max_count, 'bought_count': bought_count}
    item_behavior_count = train_user.find(
        {"user_id": user_id, "item_id": item_id, "time": {'$gt': starttime, "$lt": stoptime}}).count()
    if max_count == 0:
        return 0
    return float(item_behavior_count) / float(max_count)


@DeprecationWarning
def cal_positive_userset_vecvalues(fin_path='../data/positive_userset_2015-04-12-14-32-11.csv',
                                   fout_path='../data/popularity_desire_behaviorRate_data.csv'):
    """
    计算剩下的3个维度的值（商品热门度、用户购买欲、操作比重），并保存在csv文件中
    格式：[user_id]_[item_id],popularity,desire,behavior_rate
    :param fin_path: 正样本训练集的csv数据文件
    :param fout_path: 结果输出路径
    :return:
    """
    fin = open(fin_path, 'r')
    fout = open(fout_path, 'w')
    logger.info('cal_positive_userset_vecvalues start')
    fout.write('user_id_item_id,popularity,desire,behavior_rate\n')
    for line in fin:
        line = line.replace('\n', '')
        ids = line.split(',')
        user_id = ids[0]
        desire = cal_user_desire(user_id)
        for index in range(1, len(ids)):
            item_id = ids[index]
            fout.write(user_id + '_' + item_id + ',')
            popularity = cal_item_popularity(item_id)
            behavior_rate = cal_useritem_behavior_rate(user_id, item_id)
            datastr = '%s,%s,%s\n' % (popularity, desire, behavior_rate)
            fout.write(datastr)
    logger.info('cal_positive_userset_vecvalues done,output path=' + fout_path)


@Timer
def cal_user_behavior(connect,
                      timerange,
                      f_train_set='%s/train_set.csv' % (data_path)):
    """
    计算时间加权后的用户行为

    结果输出文件的格式：
         -------------- content ---------------
        | user_id,item_id,see,favorite,cart,buy,tag |
         --------------------------------------
        其中，see代表浏览的时间加权结果，favorite代表收藏,cart代表添加到购物车,favorite代表买

    Args:
        connect: MySQLdb.connect(), 数据库连接句柄
        timerange: 时间筛选条件
        f_train_set: string, 训练集结果文件
                 ------ content ------
                | user_id,item_id,tag |
                 ---------------------
    Returns:
        f_output: sting，输出文件
    """
    import arrow
    from math import exp

    f_output = f_train_set.replace('.csv', '_calUserBehavior.csv')  # 输出文件的名称
    predict_timestamp = arrow.get('2014-12-19').timestamp
    time_atten = 3600 * 48  # 时间戳的衰减因子, exp(-1/a * delta_t)
    (timerange_start, timerange_end) = map(lambda elem: arrow.get(elem).timestamp, timerange)
    cursor = connect.cursor()

    with open(f_train_set, 'r') as fin, open(f_output, 'w') as fout:
        fin.readline()  # 忽略首行
        fout.write('user_id,item_id,see,favorite,cart,buy,tag\n')
        counter = 0  # for log
        logger.debug('start generate...')
        for in_line in fin:
            in_cols = in_line.strip().split(',')
            [user_id, item_id, tag] = in_cols
            sql = 'select behavior_type, time from train_user where user_id=%s and item_id=%s and time>%s and time<=%s;' % (
                user_id, item_id, timerange_start, timerange_end)
            # logger.debug('sql: %s' % (sql))
            cursor.execute(sql)
            result = cursor.fetchall()
            time_weights = [0.0, 0.0, 0.0, 0.0]
            for [behavior_type, timestamp] in result:
                time_weights[int(behavior_type) - 1] += exp((timestamp - predict_timestamp) / time_atten)
            fout.write('%s,%s,%s,%s,%s,%s,%s\n' % (
                user_id, item_id, time_weights[0], time_weights[1], time_weights[2], time_weights[3], tag))
            counter += 1
            if counter % 300 == 0:
                logger.debug('NO.%s: user_id=%s, item_id=%s, time_weights=%s, tag=%s' % (
                    counter, user_id, item_id, time_weights, tag))

    cursor.close()
    return f_output


@Timer
def cal_vecvalues_tail(mongo_train_user_collection, fin_path='../data/train_set.csv',
                       fout_path='../data/vecvalues_tail.csv', timerange=('2014-12-12', '2014-12-19')):
    """
    计算后三维的向量，需要mongodb支持
    :param fin_path:样本集csv路径
    :param fout_path:结果路径
    :param stoptime:计算的截止日期，格式'2014-12-19'
    :return:
    """
    logger.info('cal_vecvalues_tail start')
    fin = open(fin_path, 'r')
    fin.readline()  # 跳过标题行
    fout = open(fout_path, 'w')
    fout.write('tag,popularity,desire,behavior_rate\n')
    count = 0
    # stoptime += ' 00'
    for line in fin:
        line = line.replace('\n', '')
        data = line.split(',')
        user_id = data[0]
        item_id = data[1]
        tag = data[2]
        popularity = cal_item_popularity(mongo_train_user_collection, item_id, timerange=timerange)
        desire = cal_user_desire(mongo_train_user_collection, user_id, timerange=timerange)
        behavior_rate = cal_useritem_behavior_rate(mongo_train_user_collection, user_id, item_id, timerange=timerange)
        datastr = '%s,%s,%s,%s\n' % (tag, popularity, desire, behavior_rate)
        # datastr = tag + ',' + str(popularity) + ',' + str(desire) + ',' + str(behavior_rate) + '\n'
        fout.write(datastr)
        count += 1
        if count % 5000 == 0:
            logger.info('calculated count:\t%s' % count)
    logger.info('cal_vecvalues_tail done, result path=' + fout_path)


@Timer
def combine_data(userbehavior_filepath='%s/train_set_calUserBehavior.csv' % (data_path),
                 tail_filepath='%s/vecvalues_tail.csv' % (data_path),
                 csv_output_path='%s/combined_vec_data.csv' % (data_path),
                 svm_output_path='%s/svmdata.dat' % (data_path)):
    """
    组合前四维向量与后三维向量数据
    :param userbehavior_filepath:
    :param tail_filepath:
    :param csv_output_path:
    :param svm_output_path:
    :return:
    """

    logger.info('start combining data')
    userbehavior_file = open(userbehavior_filepath, 'r')
    tail_file = open(tail_filepath, 'r')
    csvout = open(csv_output_path, 'w')
    csvout.write('tag,see,favorite,cart,buy,popularity,desire,behavior_rate\n')
    svmout = open(svm_output_path, 'w')
    behaviors = userbehavior_file.readlines()
    tails = tail_file.readlines()
    for index in range(1, len(behaviors)):
        tails_line = tails[index].replace('\n', '')
        data_tail = tails_line.split(',')
        behavior_line = behaviors[index].replace('\n', '')
        data_behavior = behavior_line.split(',')
        tag = data_tail[0]
        see = data_behavior[2]
        favorite = data_behavior[3]
        cart = data_behavior[4]
        buy = data_behavior[5]
        popularity = data_tail[1]
        desire = data_tail[2]
        behavior_rate = data_tail[3]
        # 构建svm用向量数据格式
        svmstr = '%s 0:%s 1:%s 2:%s 3:%s 4:%s 5:%s 6:%s\n' % (
            tag, see, favorite, cart, buy, popularity, desire, behavior_rate)
        csvstr = '%s,%s,%s,%s,%s,%s,%s,%s\n' % (
            tag, see, favorite, cart, buy, popularity, desire, behavior_rate)
        svmout.write(svmstr)
        csvout.write(csvstr)
    logger.info('combine_data done,svmout_path=%s, csvout_path=%s' % (svm_output_path, csv_output_path))


@Timer
def get_predict_vecdata(set_timerange=('2014-12-18', '2014-12-19'),
                        behavior_timerange=('2014-12-12', '2014-12-19'),
                        predict_set_path='%s/predict/predict_set.csv' % (data_path),
                        predict_vectail_path='%s/predict/predict_vectail.csv' % (data_path),
                        csv_output_path='%s/predict/combined_vec_data.csv' % (data_path),
                        svm_output_path='%s/predict/svmdata.dat' % (data_path)):
    """
    生成预测集，需要制定时间范围与各输出路径
    :param set_timerange: 潜在购买商品的时间范围
    :param behavior_timerange: 计算向量值时的时间范围
    """

    from data_preprocess import generate_userset
    import MySQLdb
    from data_preprocess.MongoDB_Utils import MongodbUtils

    logger.info('start get_predict_vecdata, set_timerange=%s to %s,  behavior_timerange = %s to %s' % (
        set_timerange[0], set_timerange[1], behavior_timerange[0], behavior_timerange[1]))
    connect = MySQLdb.connect(host='10.108.192.119',
                              user='tianchi_data',
                              passwd='tianchi_data',
                              db='tianchi')

    mongo_utils = MongodbUtils(db_address, 27017)
    train_user = mongo_utils.get_db().train_user
    # predict_set_path = '%s/temp/predict_set.csv' % (data_path)
    generate_userset.generate_predict_set(connect, set_timerange, predict_set_path)
    # predict_vectail_path = '%s/temp/predict_vectail.csv' % (data_path)
    # stoptime = timerange[1]
    cal_vecvalues_tail(train_user, predict_set_path, predict_vectail_path, behavior_timerange)
    predict_vecbehavior_path = predict_set_path.replace('.csv', '_calUserBehavior.csv')
    cal_user_behavior(connect, behavior_timerange, predict_set_path)
    combine_data(predict_vecbehavior_path, predict_vectail_path, csv_output_path, svm_output_path)


@Timer
def get_train_vecdata(train_set_path='%s/train/train_set.csv' % data_path,
                      combined_out_path='%s/train/combined_out.csv' % data_path,
                      svmdata_out_path='%s/train/svmdata.dat' % data_path,
                      set_timerange=('2014-12-18', '2014-12-19'),
                      behavior_timerange=('2014-12-12', '2014-12-19')):
    """
    生成训练数据集
    """

    import MySQLdb
    from data_preprocess import generate_userset
    from data_preprocess.MongoDB_Utils import MongodbUtils

    logger.info('start get_train_vecdata, timerange=%s to %s' % (set_timerange[0], set_timerange[1]))
    connect = MySQLdb.connect(host='10.108.192.119',
                              user='tianchi_data',
                              passwd='tianchi_data',
                              db='tianchi')

    mongo_utils = MongodbUtils(db_address, 27017)
    train_user = mongo_utils.get_db().train_user
    # generate_userset.generate_train_set(connect, ('2014-12-18', '2014-12-19'), ('2014-12-18', '2014-12-19'),
    # r'../data/train/train_set_1819.csv')
    generate_userset.generate_train_set(connect, set_timerange, set_timerange,
                                        train_set_path)
    vectail_path = train_set_path.replace('.csv', '_vectail.csv')
    cal_vecvalues_tail(train_user, train_set_path, vectail_path, behavior_timerange)
    # predict_vecbehavior_path = predict_set_path.replace('.csv', '_calUserBehavior.csv')
    cal_user_behavior_path = train_set_path.replace('.csv', '_calUserBehavior.csv')
    cal_user_behavior(connect, behavior_timerange, train_set_path)
    combine_data(cal_user_behavior_path,
                 vectail_path,
                 combined_out_path,
                 svmdata_out_path)


if __name__ == '__main__':
    # cal_vecvalues_tail()

    # combine_data()

    # **************************************************

    # 生成预测集数据
    predict_path = '%s/predict_1219_timerange' % data_path
    get_predict_vecdata(set_timerange=('2014-12-18', '2014-12-19'),
                        behavior_timerange=('2014-12-14', '2014-12-19'),
                        predict_set_path='%s/predict_1219_set.csv' % predict_path,
                        predict_vectail_path='%s/predict_1219_vectail.csv' % predict_path,
                        csv_output_path='%s/predict_1219_combined.csv' % predict_path,
                        svm_output_path='%s/predict_1219_svmdata.dat' % predict_path)

    # **************************************************

    # 生成测试集数据
    test_path = '%s/test_1205_timerange' % data_path
    get_predict_vecdata(set_timerange=('2014-12-04', '2014-12-05'),
                        behavior_timerange=('2014-11-30', '2014-12-05'),
                        predict_set_path='%s/test_1205_set.csv' % test_path,
                        predict_vectail_path='%s/test_1205_vectail.csv' % test_path,
                        csv_output_path='%s/test_1205_combined.csv' % test_path,
                        svm_output_path='%s/test_1205_svmdata.dat' % test_path)
    # get_predict_vecdata(set_timerange=('2014-12-04', '2014-12-05'),
    # behavior_timerange=('2014-12-01', '2014-12-05'),
    # predict_set_path='%s/test_1205/test_1205_set.csv' % data_path,
    # predict_vectail_path='%s/test_1205/test_1205_vectail.csv' % data_path,
    # csv_output_path='%s/test_1205/test_1205_combined.csv' % data_path,
    # svm_output_path='%s/test_1205/test_1205_svmdata.dat' % data_path)

    # **************************************************

    # 生成测试用训练集数据
    train_path = '%s/train_1205_timerange' % data_path
    get_train_vecdata(set_timerange=('2014-12-04', '2014-12-05'),
                      behavior_timerange=('2014-11-29', '2014-12-04'),
                      train_set_path='%s/train_set_1205.csv' % train_path,
                      combined_out_path='%s/combined_out_1205.csv' % train_path,
                      svmdata_out_path='%s/svmdata_1205.dat' % data_path)

    # 生成预测用训练集数据
    train_path = '%s/train_1219_timerange' % data_path
    get_train_vecdata(set_timerange=('2014-12-18', '2014-12-19'),
                      behavior_timerange=('2014-12-13', '2014-12-18'),
                      train_set_path='%s/train_set_1219.csv' % train_path,
                      combined_out_path='%s/combined_out_1219.csv' % train_path,
                      svmdata_out_path='%s/svmdata_1219.dat' % data_path)

    # get_train_vecdata(set_timerange=('2014-12-04', '2014-12-05'),
    #                   behavior_timerange=('2014-12-01', '2014-12-05'),
    #                   train_set_path='%s/train_1205/train_set_1205.csv' % data_path,
    #                   combined_out_path='%s/train_1205/combined_out_1205.csv' % data_path,
    #                   svmdata_out_path='%s/train_1205/svmdata_1205.dat' % data_path)




    # **********************************************************
    # 生成训练集，以12-8到12-9之间购买过的行为作为正样本，12-8以前的所有数据作为构建向量的依据
    # import MySQLdb
    # from data_preprocess import generate_userset
    #
    # connect = MySQLdb.connect(host='127.0.0.1',
    # user='tianchi_data',
    # passwd='tianchi_data',
    # db='tianchi')
    # from data_preprocess.MongoDB_Utils import MongodbUtils
    #
    # mongo_utils = MongodbUtils(db_address, 27017)
    # train_user = mongo_utils.get_db().train_user
    # generate_userset.generate_train_set(connect, ('2014-12-18', '2014-12-19'), ('2014-12-18', '2014-12-19'),
    # r'../data/train_set_1819.csv')
    # cal_vecvalues_tail(train_user, r'../data/train_set_1819.csv', r'../data/temp/tail.csv', '2014-12-18')
    # # predict_vecbehavior_path = predict_set_path.replace('.csv', '_calUserBehavior.csv')
    # cal_user_behavior(connect, ('2014-11-17', '2014-12-18'), r'../data/train_set_1819.csv')
    # combine_data(r'../data/train_set_1819_calUserBehavior.csv',
    # r'../data/temp/tail.csv',
    # r'../data/temp/combined_out.csv',
    # r'../data/temp/svm_out.csv')
    # ************************************************************



    # connect = MySQLdb.connect(host='127.0.0.1',
    # user='tianchi_data',
    # passwd='tianchi_data',
    # db='tianchi')
    # cal_user_behavior(connect)
    # connect.close()
