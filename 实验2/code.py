import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("./income_classification.csv")
print("数据的维度是：",data.ndim) # 显示数据维度
print('数据的形状是：',data.shape)
# re = data[data['workclass'] == '?'].index  # 删除异常值,
# data =data.drop(re)  # 横向删除


# print(pd.get_dummies(data['marital-status']).columns)  # 查看编码处理后有多少类
# print("处理前......\n")
# print(data[['sex','marital-status']].head(),'\n')  # 处理前
print(type(data['marital-status']))
print(data['marital-status'].shape)
# 处理marital-status  属性字符串
classMap = {'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2, 'Married-spouse-absent': 3,
            'Never-married': 4, 'Separated': 5,
            'Widowed': 6}
data['marital-status'] = data['marital-status'].map(classMap)


# 处理性别
"""classMap1 = {"Male": 1, "Female": 0}
data['sex'] = data['sex'].map(classMap1)

# 处理workclass 属性字符串
list1 = pd.get_dummies(data['workclass']).columns
classMap2 = {key: value for value, key in enumerate(list1)}  # 字典推导式
data['workclass'] = data['workclass'].map(classMap2)
print("workclass编码之后前5行的结果:\n",data['workclass'].head())
# print(pd.get_dummies(data['sex']))

# 处理age
data['age'] = pd.cut(data['age'], bins=5, retbins=True)[0]  # 等宽法进行连续数据离散化
list1 = pd.get_dummies(data['age']).columns
classMap = {key:value for value,key in enumerate(list1)}
data['age'] = data['age'].map(classMap)
print("年龄离散化之编码后前五行的结果为:\n",data['age'].head())

# 处理分类标签 0 = <=50K 1 = >50K
list2 = pd.get_dummies(data['income']).columns
classMap3 = {key: value for value, key in enumerate(list2)}
data['income'] = data['income'].map(classMap3)


# print("处理后......\n")
# print(data[['sex','marital-status']].head())
target = data['income']
data = data[['workclass', 'sex', 'marital-status', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]

# 划分数据集
data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=0.7)


tree1 = DecisionTreeClassifier(criterion='gini', max_depth=10)
tree1 = tree1.fit(data_train , target_train)
accuracy_cart = tree1.score(data_test, target_test)
print('CART树分类准确率：',accuracy_cart)
#pred = tree.predict(data_test) # 预测结果
#true = np.sum(pred == target_test)
#print(true/data_test.shape[0])


ense = RandomForestClassifier(max_depth=10)
ense = ense.fit(data_train , target_train)
accuracy_random = ense.score(data_test, target_test)
print('随机森林分类准确率：',accuracy_random)
#pred1 = ense.predict(data_test)
#true1 = np.sum(pred1 == target_test)
#print(true1/data_test.shape[0])"""