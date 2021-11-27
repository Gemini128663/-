"""import pandas as pd
import numpy as np
from pyspark.mllib.recommendation import ALS
from sklearn.model_selection import train_test_split

data = np.loadtxt('ratings copy.csv', skiprows=1, delimiter=',')

train, test = train_test_split(data,train_size=0.8)
train, val = train_test_split(train,train_size=0.75)
print(type(train))

A = np.array([[3, 1, 0], [-1, 2, 1], [3, 4, 2]])
print("A的秩为{}".format(np.linalg.matrix_rank(train))) #计算矩阵A的秩"""

import numpy as np
import pandas as pd

# B = np.array([[3, 1, 0], [3, 2, 1], [3, 3, 2], [4, 1, 1], [4, 2, 2]])
# b = B[:, 0].map(lambda x: (x[1], x[2]))

data = pd.read_csv('ratings copy.csv', header=None, names=['id', 'name', 'score', 'time'])

# print(data)
dict1 = {}
for name, data in data.groupby('id'):
    dict1[name] = {key: value for key, value in zip(data['name'], data['score'])}
print(len(dict1))
print(dict1)
#print(dict1)
"""
user = {
    '3':{"1":"0","2":"1","3":"2"},
    '4':{"1":"1","2":"2"}
}
"""
