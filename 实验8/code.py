# coding = utf-8
# @Time : 2021/11/27 17:51
# @Author : 思珂
# @File : test.py
# @Software: PyCharm
# !/usr/bin/python
# -*- coding: UTF-8 -*-
'''
基于用户的推荐算法
'''
from math import sqrt, pow
import operator
import pandas as pd


class UserCf:

    # 获得初始化数据
    def __init__(self, data):
        self.data = data

    # 计算两个用户的皮尔逊相关系数
    def pearson(self, user1, user2):  # 数据格式为：电影，评分
        sumXY = 0.0
        n = 0
        sumX = 0.0
        sumY = 0.0
        sumX2 = 0.0
        sumY2 = 0.0
        try:
            for movie1, score1 in user1.items():
                if movie1 in user2.keys():  # 计算公共的电影的评分
                    n += 1
                    sumXY += score1 * user2[movie1]
                    sumX += score1
                    sumY += user2[movie1]
                    sumX2 += pow(score1, 2)
                    sumY2 += pow(user2[movie1], 2)

            molecule = sumXY - (sumX * sumY) / n
            denominator = sqrt((sumX2 - pow(sumX, 2) / n) * (sumY2 - pow(sumY, 2) / n))

            r = round(molecule / denominator, 2)
        except Exception as e:
            # print("异常信息:", e)
            return 0

        return r

    # 计算与当前用户的距离，获得最临近的用户
    def nearstUser(self, username, n):
        distances = {}  # 用户，相似度
        for otherUser, items in self.data.items():  # 遍历整个数据集
            if otherUser not in username:  # 非当前的用户
                distance = self.pearson(self.data[username], self.data[otherUser])  # 计算两个用户的相似度
                distances[otherUser] = distance

        sortedDistance = sorted(distances.items(), key=operator.itemgetter(1), reverse=True)  # 最相似的N个用户

        print("排序后的用户为：", sortedDistance[:n])  # 相关系数最大的前n个用户
        return sortedDistance[:n]

    # 给用户推荐电影
    def recomand(self, username, n):
        recommand = {}  # 待推荐的电影
        for user, score in dict(self.nearstUser(username, n)).items():  # 最相近的n个用户
            print("推荐的用户：", (user, score))
            for movies, scores in self.data[user].items():  # 推荐的用户的电影列表

                if movies not in recommand.keys():  # 添加到推荐列表中
                    recommand[movies] = scores

        return sorted(recommand.items(), key=operator.itemgetter(1), reverse=True)  # 对推荐的结果按照电影评分排序


if __name__ == '__main__':
    data = pd.read_csv('ratings copy.csv', header=None, names=['id', 'name', 'score', 'time'])
    num = eval(input("请输入要推荐的人数数量: "))
    dict1 = {}
    for name, data in data.groupby('id'):
        name = str(name)
        dict1[name] = {str(key): value for key, value in zip(data['name'], data['score'])}

    userCf = UserCf(data=dict1)
    recommandList = userCf.recomand('67', num)
    print("最终推荐的10部电影:", recommandList[:10])
    # print(recommandList[:10])
