# coding=utf-8
import json
import os
import sys
import MySQLdb

project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
data_path = '%s/data' % (project_path)

# project import
sys.path.append(project_path)
from log.get_logger import logger, Timer

__author__ = 'Jayvee'


class PopularityInCategory:
    def __init__(self, train_user_connect, stoptime_str):
        from datetime import datetime

        self.train_user = train_user_connect
        self.stoptime_str = stoptime_str
        self.stoptime = datetime.strptime(str(self.stoptime_str + ' 00'), '%Y-%m-%d %H')
        self.category_bought_dict = {}
        self.item_bought_dict = {}
        self.result_dict = {}

    def get_popularity_in_category(self, item_id):
        """
        计算某个商品在其所属的类内的热门程度，若在商品子集（train_item中）不存在该商品，则返回0，
        否则返回被购买数的百分比
        :param item_id:商品id
        :param stoptime_str:截止日期
        :param train_user_connect: Mongodb的train_user表连接
        :return:
        """
        if self.result_dict.has_key(item_id):
            return self.result_dict[item_id]
        else:
            current_item = self.train_user.find_one({'item_id': item_id})
            category_id = current_item['item_category']
            # if self.item_bought_dict.has_key(item_id):
            # item_bought_count = self.item_bought_dict[item_id]
            # else:
            item_bought_count = self.train_user.find({'item_id': item_id,
                                                      'behavior_type': '4',
                                                      'time': {'$lt': self.stoptime}}).count()
            # self.item_bought_dict[item_id] = item_bought_count
            if item_bought_count != 0:
                if self.category_bought_dict.has_key(category_id):
                    bought_max_count = self.category_bought_dict[category_id]
                else:
                    bought_max_count = self.train_user.find({'item_category': category_id,
                                                             'behavior_type': '4',
                                                             'time': {'$lt': self.stoptime}}).count()
                    # bought_max_count = temp.count()
                    # k = len(temp.distinct('item_id'))
                self.category_bought_dict[category_id] = bought_max_count
                if bought_max_count != 0:
                    popularity_in_category = float(item_bought_count) / bought_max_count
                    logger.debug('item ' + item_id + ' popularity_in_category = ' + str(popularity_in_category))
                    self.result_dict[item_id] = popularity_in_category
                    return popularity_in_category
                else:
                    self.result_dict[item_id] = 0.0
                    logger.debug('item ' + item_id + ' popularity_in_category = ' + str(0.0))
                    return 0.0
            else:
                logger.debug('item ' + item_id + ' popularity_in_category = ' + str(0.0))
                return 0.0


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
def find_category_relationship(train_user_connect, train_item_connect,
                               json_output_path=r'%s/relationData_new.json' % data_path,
                               csv_output_path=r'%s/relationData_new.csv' % data_path,
                               time_window=1):
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


@Timer
def generate_from_popularity_in_category(f_recommend, stoptime_str, train_user_connect):
    """
    根据类内排名生成结果

    Args:
        f_recommend: fin, 推荐结果文件
        stoptime_str: string, 预测日期
        train_user_connect: MongoDB的 train_user的连接
    Returns:
        f_output: fout, 最终输出结果
    """
    import random
    category_popularity_item = dict()  # key:item_id, value:类内流行度
    category_popularity = dict()  # key:(user_id,item_id), value:类内流行度

    with open(f_recommend, 'r') as fin:
        fin.readline()
        for line in fin:
            cols = line.strip().split(',')
            ui_tuple = (cols[0], cols[1])
            if cols[1] not in category_popularity_item:
                category_popularity[ui_tuple] = cal_popularity_in_category(cols[1], stoptime_str, train_user_connect)
            else:
                category_popularity[ui_tuple] = category_popularity_item[cols[1]]

    # 最终结果由类内排名前%25的 加上随机的25%
    sorted_result = sorted(category_popularity.iteritems(), key=lambda d: d[1], reverse=True)
    index_slice = int(0.25*len(sorted_result))
    front_25 = sorted_result[:index_slice]
    last_75 = sorted_result[index_slice:]
    random.shuffle(last_75)
    random_25 = last_75[:index_slice]

    # 写出最后结果
    f_output = f_recommend.replace('.csv', '_categoryPopularity.csv')
    logger.debug('Start store result to %s' % (f_output))
    with open(f_output, 'w') as fout:
        fout.write('user_id,item_id\n')
        for elem in front_25:
            logger.debug('front 25, popularity:%s' % (elem[1]))
            fout.write('%s,%s\n' % (elem[0][0], elem[0][1]))
        for elem in random_25:
            logger.debug('random 25, popularity:%s' % (elem[1]))
            fout.write('%s,%s\n' % (elem[0][0], elem[0][1]))

    return f_output


@Timer
def filter_with_category_popularity(connect, train_user_connect, f_recommend, f_category_relationship, stoptime_str):
    """
    用类间承接关系和类内排名过滤结果

    Args:
        connect: MySQLdb.connect(), 数据库连接句柄
        train_user_connect: Mongodb的train_user表连接
        f_recommend: fin, 推荐结果文件
        f_category_relationship: fin, 类间承接关系
        stoptime_str: string, 截止日期
    Returns:
        f_output: fout, 过滤后的结果 
    """
    import arrow
    import random

    cursor = connect.cursor()
    f_output = f_recommend.replace('.csv', '_filter.csv')
    logger.debug('Start filter recommend result..')
    
    # 根据推荐文件生成
    stoptime_timestamp = arrow.get(stoptime_str).timestamp
    recommend_dict = {}  # {u_id1:[i_id1,i_id2], u_id2:[i_id3,i_id4]}
    with open(f_recommend, 'r') as fin:
        fin.readline()
        for line in fin:
            cols = line.strip().split(',')
            if recommend_dict.has_key(cols[0]):
                recommend_dict[cols[0]].append(cols[1])
            else:
                recommend_dict[cols[0]] = [cols[1]]
    logger.debug('完成根据推荐文件生成第一步, len:%s' % (len(recommend_dict)))
    # 分两步生成最后的dict是为了减少mysql查询数
    recommend_tuple_dict = {}  # {(u_id1,i_id1):(u_last_category, id1_category), (u_id1, i_id2):(u_last_category, id2_category)}
    user_counter = 0
    for (u_id, i_ids) in recommend_dict.iteritems():
        sql = 'SELECT item_category FROM train_user WHERE user_id=%s and time<%s ORDER BY time DESC limit 1;' % (u_id, stoptime_timestamp)
        cursor.execute(sql)
        result = cursor.fetchall()
        user_last_category = result[0][0]
        user_counter += 1
        if user_counter % 200 == 0:
            logger.debug('No.%s user, user_id=%s, last_item_category=%s' % (user_counter, u_id, user_last_category))
        for i_id in i_ids:
            sql = 'SELECT item_category FROM train_item WHERE item_id=%s;' % (i_id)
            cursor.execute(sql)
            result = cursor.fetchall()
            recommend_tuple_dict[(u_id, i_id)] = (user_last_category, result[0][0])
    logger.debug('原推荐结果长度:%s' % (len(recommend_tuple_dict)))

    # 根据承接关系文件生成
    relationship_set = set()
    with open(f_category_relationship, 'r') as fin:
        fin.readline() 
        for line in fin:
            cols = line.strip().split(',')
            relationship_set.add((cols[0], cols[1]))
    logger.debug('承接关系结果长度:%s' % (len(relationship_set)))

    # 输出结果
    with open(f_output, 'w') as fout:
        in_counter = 0
        random_counter = 0
        fout.write('user_id,item_id\n')
        for ((user_id, item_id), category_tuple) in recommend_tuple_dict.iteritems():
            if category_tuple in relationship_set:
                in_counter += 1
                fout.write('%s,%s\n' % (user_id, item_id))
            else:
                if random.random() <= cal_popularity_in_category(item_id, stoptime_str, train_user_connect):
                    random_counter += 1
                    fout.write('%s,%s\n' % (user_id, item_id))
                    logger.debug('NO.%s random pick, [%s,%s]' % (random_counter, user_id, item_id))

    logger.info('对推荐结果的筛选完成，结果路径:%s' % ())
    logger.info('筛选前%s, 筛选后%s. 其中在承接关系的有%s, 随机挑选的有%s' %(len(recommend_tuple_dict), in_counter+random_counter, in_counter, random_counter))
        


@Timer
def filter_with_mongodb(train_user_connect, f_recommend_path, f_category_relation_path,
                        fout_path='%s/result/filter_output.csv' % data_path,
                        time_range=('2014-12-04', '2014-12-05')):
    """
    推荐结果筛选的mongoDB版
    :param train_user_connect:mongodb的train_user连接
    :param f_recommend_path:推荐结果的csv文件路径
    :param f_category_relation_path:精简后的类别关系csv路径
    :param fout_path:筛选结果输出路径
    :param time_range:“最近购买”的时间范围
    :return:
    """
    import pymongo
    import random
    import math

    logger.info('start filter_with_mongoDB')
    popularity_calculator = PopularityInCategory(train_user_connect, time_range[1])
    # 推荐列表预处理
    f_recommend = open(f_recommend_path)
    f_recommend.readline()  # 忽略首行
    recommend_dict = {}
    old_count = 0
    new_count = 0
    for line in f_recommend:
        cols = line.strip().split(',')
        user_id = cols[0]
        item_id = cols[1]
        if recommend_dict.has_key(user_id):
            recommend_dict[user_id].append(item_id)
        else:
            recommend_dict[user_id] = [item_id]
            # link_count = cols[2]
    # 类别关系列表预处理
    f_category_relation = open(f_category_relation_path)
    f_category_relation.readline()
    category_dict = {}
    for line in f_category_relation:
        cols = line.strip().split(',')
        source = cols[0]
        target = cols[1]
        if category_dict.has_key(source):
            category_dict[source].append(target)
        else:
            category_dict[source] = [target]
    # 开始进行筛选
    logger.debug('start filtering!')
    fout = open(fout_path, 'w')
    fout.write('user_id,item_id\n')
    result = []
    usercount = 0
    for u_id in recommend_dict.keys():
        usercount += 1
        logger.debug(usercount)
        i_ids = recommend_dict[u_id]
        last_buy_cursor = train_user_connect.find({'user_id': u_id,
                                                   'behavior_type': '4',
                                                   'time': {
                                                       '$gt': time_range[0],
                                                       '$lt': time_range[1]}}).sort('time',
                                                                                    pymongo.DESCENDING)
        item_category_dict = {}
        item_popularity_in_category_dict = {}
        last_buy = next(last_buy_cursor, None)
        for i_id in i_ids:
            old_count += 1
            if item_category_dict.has_key(i_id):
                i_category = item_category_dict.get(i_id)
                temp_category = i_category
            else:
                temp = train_user_connect.find_one({'item_id': i_id})
                temp_category = temp['item_category']
                item_category_dict[i_id] = temp_category
            # 判断所推荐的商品种类是否与该用户最后一次购买的类别存在顺承关系
            if last_buy and temp_category in category_dict[last_buy['item_category']]:
                # last_category = last_buy['item_category']
                # if temp_category in category_dict[last_category]:
                logger.debug('有承接关系 u_id=%s,i_id=%s' % (u_id, i_id))
                fout.write('%s,%s\n' % (u_id, i_id))
                result.append((u_id, i_id))
                new_count += 1
            else:  # 不符合顺承关系则进入随机保留
                if item_popularity_in_category_dict.has_key(i_id):
                    item_popularity = item_popularity_in_category_dict[i_id]
                else:
                    item_popularity = popularity_calculator.get_popularity_in_category(i_id)
                    item_popularity_in_category_dict[i_id] = item_popularity

                item_popularity = 1 / (1 + math.e ** (item_popularity +1))
                logger.debug('random prob = %s' % item_popularity)
                if random.random() <= item_popularity:
                    fout.write('%s,%s\n' % (u_id, i_id))
                    result.append((u_id, i_id))
                    new_count += 1

            if old_count % 10 == 0:  # logger哨兵
                logger.debug('No.%s origin recommend filtered' % old_count)
    logger.info('done! origin recommend num is %s, current recommend num is %s\nresult output path: %s' % (
        old_count, new_count, fout_path))
    return result


if __name__ == '__main__':
    from data_preprocess.MongoDB_Utils import MongodbUtils

    db_address = json.loads(open('%s/conf/DB_Address.conf' % (project_path), 'r').read())['MongoDB_Address']

    mongo_utils = MongodbUtils(db_address, 27017)
    train_user = mongo_utils.get_db().train_user
    train_item = mongo_utils.get_db().train_item
    #train_user = mongo_utils.get_db().train_user_new
    #train_item = mongo_utils.get_db().train_item_new

    #find_category_relationship(train_user, train_item, '%s/relationDict.json' % data_path, 3)
    #f_recommend = '%s/test_1206/RandomForest_recommend_intersect.csv' % (data_path)
    #generate_from_popularity_in_category(f_recommend, '2014-12-06', train_user)

    """
    # find_category_relationship(train_user, train_item, json_output_path='%s/relationDict.json' % data_path,
    # csv_output_path='%s/relationDict.csv' % data_path)
    find_category_relationship(train_user, train_item,time_window=1)

    # 类内热门度调用示例
    # print cal_popularity_in_category('166670035', '2014-12-19', train_user)
    """

    # 对结果进行筛选
    # connect = MySQLdb.connect(host='127.0.0.1',
    # user='tianchi_data',
    # passwd='tianchi_data',
    # db='tianchi')
    # f_recommend = '%s/predict_1219/RandomForest_recommend_intersect.csv' % (data_path)
    # filter_with_category_popularity(connect, train_user, f_recommend, f_category_relationship, '2014-12-19')
    f_recommend = '%s/test_1205/RandomForest_recommend_intersect.csv' % (data_path)
    f_category_relationship = '%s/relationship_1.csv' % (data_path)
    result_path = f_recommend.replace('.csv', '_mongodb_filter.csv')
    # filter_with_category_popularity(connect, train_user, f_recommend, f_category_relationship, '2014-12-06')
    filter_with_mongodb(train_user, f_recommend, f_category_relationship, fout_path=result_path,
                        time_range=('2014-12-4', '2014-12-05'))
