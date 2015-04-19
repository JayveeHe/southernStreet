# coding=utf-8
import json
import os
import MySQLdb
import sys
from data_preprocess.MongoDB_Utils import MongodbUtils
from log.get_logger import logger, Timer

__author__ = 'Jayvee'


def popularity_in_category(item_id, stoptime_str, train_user_connect, train_item_connect):
    """
    计算某个商品在其所属的类内的热门程度
    :param item_id:
    :param stoptime_str:
    :param train_user_connect:
    :param train_item_connect:
    :return:
    """
    from datetime import datetime

    stoptime = datetime.strptime(str(stoptime_str + ' 00'), '%Y-%m-%d %H')
    category_id = train_item_connect.find({'item_id': item_id})['item_category']
    itemids = train_item_connect.find({'item_category': category_id})
    bought_max_count = 0
    item_bought_count = 0
    itemDict = {}
    itemList = []
    for doc in itemids:
        itemid = doc['item_id']
        bought_count = train_user_connect.find({'item_id': itemid,
                                                'behavior_type': '4',
                                                'time': {'$lt': stoptime}}).count()
        itemDict[itemid] = bought_count
        itemList.append((itemid, bought_count))
        bought_max_count += bought_count
        if itemid == item_id:
            item_bought_count = bought_count

    popularity_in_category = float(item_bought_count) / bought_max_count
    logger.debug('item ' + item_id + ' popularity_in_category = ' + str(popularity_in_category))
    return popularity_in_category


@Timer
def find_category_relationship(train_user_connect, train_item_connect, time_window=2):
    """
    计算商品子集中所有类别的承接关系
    :param train_user_connect:
    :param train_item_connect:
    :param time_window:
    :return:
    """
    import pymongo
    import json

    userids = train_user_connect.distinct('user_id')

    relationDict = {}
    itemcount = 0
    usercount = 0
    output = open('../data/relationData.json', 'w')
    for user_id in userids:
        usercount += 1
        print 'user_index:'+str(usercount)
        # 返回根据时间升序排序的所有该用户的购买行为
        user_buy_behaviors = train_user_connect.find({'user_id': user_id,
                                                      'behavior_type': '4'}).sort('time', pymongo.ASCENDING)
        buyList = []
        for buy_behavior in user_buy_behaviors:
            buyList.append(buy_behavior)
        # 根据时间窗口寻找类别之间的承接关系
        len_buylist = len(buyList)
        print 'len_buylist = '+str(len_buylist)
        logger.debug('user_index:'+str(usercount)+'\tlen_buylist = '+str(len_buylist))
        for i in range(len_buylist):
            currentBuy = buyList[i]
            itemcount += 1
            currentItem = train_item_connect.find_one({'item_id': currentBuy['item_id']})
            if currentItem != None:
                # for currentItem in currentItem_cursor:
                # if type(currentItem) != None:
                # print len(currentItem)
                currentCategory = currentItem['item_category']
                targetCategoryDict = {}
                # if currentCategory != None:
                if relationDict.has_key(currentCategory):
                    targetCategoryDict = relationDict.get(currentCategory)
                else:
                    relationDict[currentCategory] = targetCategoryDict
                # else:
                # continue  # 商品子集中没有该商品，则跳过
                j = i
                while j < len_buylist:
                    if (buyList[j]['time'] - currentBuy['time']).days <= time_window:
                        # 两次购买行为在时间窗口tw内，则存在承接关系
                        targetItem = train_item_connect.find_one({'item_id': buyList[j]['item_id']})
                        if targetItem is not None and targetItem['item_category'] != currentCategory:
                            targetCategory = targetItem['item_category']
                            # 更新dict中的次数计数
                            if targetCategoryDict.has_key(targetCategory):
                                targetCategoryDict[targetCategory] += 1
                            else:
                                targetCategoryDict[targetCategory] = 1
                        j += 1
                    else:
                        break  # 若购买行为超出了时间窗口，则跳出while
                        # break

    jsonstr = json.dumps(relationDict)

    output.write(jsonstr)


if __name__ == '__main__':
    project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    data_path = '%s/data' % (project_path)

    # project import
    sys.path.append(project_path)
    # connect = MySQLdb.connect(host='127.0.0.1',
    #                           user='tianchi_data',
    #                           passwd='tianchi_data',
    #                           db='tianchi')
    db_address = json.loads(open('%s/conf/DB_Address.conf' % (project_path), 'r').read())['MongoDB_Address']

    mongo_utils = MongodbUtils(db_address, 27017)
    train_user = mongo_utils.get_db().train_user
    train_item = mongo_utils.get_db().train_item
    find_category_relationship(train_user, train_item, 3)