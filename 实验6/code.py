from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import classification_report # 分类模型评价报告
import pandas as pd
import re
import numpy as np

print("1.读取CSV文件数据中......")
data = pd.read_csv('./messages.csv')

target = data['Spam']
data = data['Subject']
list1 = []


def tokenize(message):
    message = message.lower()
    all_words = re.findall("[a-z0-9']+", message)
    s1 = set(all_words)
    list2 = list(s1.intersection(s1))  # 集合转列表
    return list2


for i in data:
    data = tokenize(i)
    list1.append(data)

"""s2 = []

for line in list1:
    list3 = []
    for word in line:
        list3.append(word)
    temps = ' '.join(list3)
    s2.append(temps)
list1 =s2"""


def createVocabList(list1):
    """创建词汇表"""
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in list1:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


print('2.构建词汇表中......')
vocabList = createVocabList(list1)


def generateMat(data, word_dict):
    """生成特征矩阵和分类矩阵"""
    num_samples = len(data)
    num_features = len(word_dict)
    feature = np.zeros((num_samples, num_features))
    classify = np.zeros(num_samples)
    for i in range(num_samples):
        data_row = data[i]
        classify[i] = target[i]
        for word in data_row:
            if word in word_dict:
                feature[i][word_dict.index(word)] = 1
    return feature, classify


print('3.生成feature矩阵和classify矩阵中......')
# print('训练矩阵特征维度:',generateMat(data,vocabList))

data_train, data_test, target_train, target_test = train_test_split(generateMat(list1, vocabList)[0],
                                                                    generateMat(list1, vocabList)[1], train_size=0.75)
print('4.根据训练数据，生成feature矩阵和classify矩阵中......')
print('训练矩阵特征维度:', data_train.shape)
print('5.根据测试数据，生成feature矩阵和classify矩阵中......')
print('测试矩阵特征维度:', data_test.shape)
print('6.构建并训练模型中......')
model = MultinomialNB().fit(data_train, target_train)
print('7.用测试集进行预测中......')
pred = model.predict(data_test)
print("预测错误数据为", (pred != target_test).sum())
print("预测的准确率为:{:.2f}%".format(accuracy_score(target_test, pred) * 100))
print("预测的精确率为:{:.2f}%".format(precision_score(target_test, pred) * 100))
print("预测的召回率为:{:.2f}%".format(recall_score(target_test, pred) * 100))
print("预测的f1值为:{:.2f}".format(f1_score(target_test, pred)))
# print(classification_report(target_test,pred))
