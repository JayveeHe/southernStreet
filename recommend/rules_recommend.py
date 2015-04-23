# coding=utf-8
import json
import os
import sys

__author__ = 'Jayvee'

project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
data_path = '%s/data' % (project_path)

# project import
sys.path.append(project_path)
from data_preprocess.MongoDB_Utils import MongodbUtils
from log.get_logger import logger, Timer


@Timer
def get_cart_list(train_user_connect, timerange=('2014-12-18', '2014-12-19')):
    from datetime import datetime

    logger.info('get_cart_list start,timerange = %s to %s' % (timerange[0], timerange[1]))
    starttime = datetime.strptime(str(timerange[0]), '%Y-%m-%d')
    stoptime = datetime.strptime(str(timerange[1]), '%Y-%m-%d')
    carts = train_user_connect.find({'behavior_type': '3',
                                     'time':
                                         {'$gt': starttime, '$lt': stoptime}})
    # .distinct('user_id')
    logger.debug('database qury done')
    carts_dict = {}
    count = 0
    for doc in carts:
        user_id = doc['user_id']
        item_id = doc['item_id']
        behavior_type = doc['behavior_type']
        item_category = doc['item_category']
        time = doc['time']
        if carts_dict.has_key(user_id):
            category_incart = carts_dict[user_id]
            if category_incart.has_key(item_category):
                category_incart[item_category].append(item_id)
            else:
                category_incart[item_category] = [item_id]
        else:
            category_incart = {item_category: [item_id]}
            carts_dict[user_id] = category_incart
        count += 1
        if count % 1000 == 0:
            logger.debug('No.%s done' % count)
    return carts_dict


@Timer
def get_buy_list(train_user_connect, timerange=('2014-12-18', '2014-12-19')):
    from datetime import datetime

    logger.info('get_buy_list start,timerange = %s to %s' % (timerange[0], timerange[1]))
    starttime = datetime.strptime(str(timerange[0]), '%Y-%m-%d')
    stoptime = datetime.strptime(str(timerange[1]), '%Y-%m-%d')
    buys = train_user_connect.find({'behavior_type': '4',
                                    'time':
                                        {'$gt': starttime, '$lt': stoptime}})
    # .distinct('user_id')
    logger.debug('database qury done')
    buy_dict = {}
    count = 0
    for doc in buys:
        user_id = doc['user_id']
        item_id = doc['item_id']
        behavior_type = doc['behavior_type']
        item_category = doc['item_category']
        time = doc['time']
        if buy_dict.has_key(user_id):
            category_inbuy = buy_dict[user_id]
            if category_inbuy.has_key(item_category):
                category_inbuy[item_category].append(item_id)
            else:
                category_inbuy[item_category] = [item_id]
        else:
            category_inbuy = {item_category: [item_id]}
            buy_dict[user_id] = category_inbuy
        count += 1
        if count % 1000 == 0:
            logger.debug('No.%s done' % count)
    return buy_dict


def determine_result(carts_dict, buys_dict):
    fresult = open('%s/result/result_rules_recomend.csv' % data_path, 'w')
    fresult.write('user_id,item_id\n')
    for user_id in carts_dict.keys():
        for cart_category in carts_dict[user_id].keys():
            if user_id in buys_dict.keys():
                if cart_category not in buys_dict[user_id].keys():
                    # 则该用户会购买该类别所有放入购物车的商品
                    for item_id in carts_dict[user_id][cart_category]:
                        fresult.write('%s,%s\n' % (user_id, item_id))
            else:
                for item_id in carts_dict[user_id][cart_category]:
                    fresult.write('%s,%s\n' % (user_id, item_id))


if __name__ == '__main__':
    db_address = json.loads(open('%s/conf/DB_Address.conf' % (project_path), 'r').read())['MongoDB_Address']

    # mongo_utils = MongodbUtils(db_address, 27017)
    # train_user = mongo_utils.get_db().train_user
    # carts_dict = get_cart_list(train_user, ('2014-12-04', '2014-12-05'))
    # open('%s/result/carts_dict_1204_1205.json' % data_path, 'w').write(json.dumps(carts_dict))
    # buys_dict = get_buy_list(train_user, ('2014-12-04', '2014-12-05'))
    # open('%s/result/buy_dict_1204_1205.json' % data_path, 'w').write(json.dumps(buys_dict))

    carts_dict = json.load(open('%s/result/carts_dict_1204_1205.json' % data_path, 'r'))
    buys_dict = json.load(open('%s/result/buy_dict_1204_1205.json' % data_path, 'r'))
    determine_result(carts_dict, buys_dict)




