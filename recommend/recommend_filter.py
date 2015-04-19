# coding=utf-8
import json
import os
import sys

project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
data_path = '%s/data' % (project_path)

# project import
sys.path.append(project_path)
from log.get_logger import logger, Timer

__author__ = 'Jayvee'


def cal_popularity_in_category(item_id, stoptime_str, train_user_connect):
    """
    计算某个商品在其所属的类内的热门程度，若在商品子集（train_item中）不存在该商品，则返回0，
    否则返回被购买数的百分比
    :param item_id:商品id
    :param stoptime_str:截止日期
    :param train_user_connect: Mongodb的train_user表连接
    :return:
    """
    from datetime import datetime

    stoptime = datetime.strptime(str(stoptime_str + ' 00'), '%Y-%m-%d %H')
    current_item = train_user_connect.find_one({'item_id': item_id})
    category_id = current_item['item_category']
    item_bought_count = train_user_connect.find({'item_id': item_id,
                                                 'behavior_type': '4',
                                                 'time': {'$lt': stoptime}}).count()
    bought_max_count = train_user_connect.find({'item_category': category_id,
                                                'behavior_type': '4',
                                                'time': {'$lt': stoptime}}).count()
    if bought_max_count != 0:
        popularity_in_category = float(item_bought_count) / bought_max_count
        logger.debug('item ' + item_id + ' popularity_in_category = ' + str(popularity_in_category))
        return popularity_in_category
    else:
        return 0.0


@Timer
def find_category_relationship(train_user_connect, train_item_connect, json_output_path='../data/relationData.json',
                               csv_output_path='../data/relationData.csv',
                               time_window=2):
    """
    计算商品子集中所有类别的承接关系
    :param train_user_connect:
    :param train_item_connect:
    :param time_window:
    :return:
    """
    import pymongo
    import json

    logger.info('find_category_relationship start!')
    userids = train_user_connect.distinct('user_id')
    logger.debug('userids loaded!')
    # category_items = train_item_connect.distinct('item_id')
    # logger.debug('category_items loaded')
    relationDict = {}
    itemcount = 0
    usercount = 0
    for user_id in userids:
        usercount += 1
        # print 'user_index:' + str(usercount)
        # 返回根据时间升序排序的所有该用户的购买行为
        user_buy_behaviors = train_user_connect.find({'user_id': user_id,
                                                      'behavior_type': '4'}).sort('time', pymongo.ASCENDING)
        categoryList = []
        # 存储（类别id，行为时间）元祖
        for buy_behavior in user_buy_behaviors:
            categoryList.append((buy_behavior['item_category'], buy_behavior['time']))
        # 根据时间窗口寻找类别之间的承接关系
        len_category = len(categoryList)
        # print 'len_buylist = ' + str(len_category)
        for i in range(len_category):
            current_item = categoryList[i]
            itemcount += 1
            currentCategory = current_item[0]
            targetCategoryDict = {}
            if relationDict.has_key(currentCategory):
                targetCategoryDict = relationDict.get(currentCategory)
            # else:
            # relationDict[currentCategory] = targetCategoryDict
            # else:
            # continue  # 商品子集中没有该商品，则跳过
            j = i
            while j < len_category:
                if (categoryList[j][1] - current_item[1]).days <= time_window:
                    # 两次购买行为在时间窗口tw内，则存在承接关系
                    if categoryList[j][0] != current_item[0]:
                        targetCategory = categoryList[j][0]
                        # 更新dict中的次数计数
                        if targetCategoryDict.has_key(targetCategory):
                            targetCategoryDict[targetCategory] += 1
                        else:
                            targetCategoryDict[targetCategory] = 1
                    j += 1
                else:
                    break  # 若购买行为超出了时间窗口，则跳出while
            if len(targetCategoryDict) > 0:
                relationDict[currentCategory] = targetCategoryDict
                # break
        if usercount % 1000 == 0:
            logger.debug('No.%s user done, user_index:%s\tlen_category = %s' % (usercount, usercount, len_category))

    jsonstr = json.dumps(relationDict)
    output = open(json_output_path, 'w')
    output.write(jsonstr)
    # dict转存为csv
    csvout = open(csv_output_path, 'w')
    csvout.write('source_category,target_category,link_count\n')
    for source in relationDict.keys():
        for target in relationDict.get(source):
            csvout.write('%s,%s,%s\n' % (source, target, relationDict[source][target]))
    logger.info('find_category_relationship done, json_output_path=%s\tcsv_output_path=%s' % (
    json_output_path, csv_output_path))


if __name__ == '__main__':
    # project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    # data_path = '%s/data' % (project_path)
    from data_preprocess.MongoDB_Utils import MongodbUtils
    # project import
    # sys.path.append(project_path)
    # connect = MySQLdb.connect(host='127.0.0.1',
    # user='tianchi_data',
    # passwd='tianchi_data',
    # db='tianchi')
    db_address = json.loads(open('%s/conf/DB_Address.conf' % (project_path), 'r').read())['MongoDB_Address']

    mongo_utils = MongodbUtils(db_address, 27017)
    train_user = mongo_utils.get_db().train_user
    train_item = mongo_utils.get_db().train_item
    # find_category_relationship(train_user, train_item, json_output_path='%s/relationDict.json' % data_path,
    # csv_output_path='%s/relationDict.csv' % data_path)
    find_category_relationship(train_user, train_item)

    # 类内热门度调用示例
    # print cal_popularity_in_category('166670035', '2014-12-19', train_user)

