import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # 线性回归函数
from sklearn.model_selection import train_test_split

train_data = pd.read_excel("./train.xlsx")
dia = train_data.iloc[:, 1:2]  # 直径
price = train_data['价格']  # 价格

"""plt.title("直径-价格 散点图")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.grid(True) # 用来显示线
plt.xlim((0,25)) # x轴范围
plt.ylim((0,25))# y轴范围
plt.scatter(dia, price, c="red")  # 绘制散点图
plt.xlabel("直径")
plt.ylabel("价格")
plt.show() # 题目一"""



"""model = LinearRegression().fit(dia, price)  # 定义模型并求解
plt.scatter(dia, price)
dia = [[0], [10], [14], [25]]
price_pre = model.predict(dia)  # 预测
plt.grid(True)  # 用来显示线
plt.xlim((0, 25))  # x轴范围
plt.ylim((0, 25))  # y轴范围
plt.plot(dia, price_pre, c='r')  # 模型拟合直线
plt.show()
alpha = model.intercept_  # 线性方程的斜率
# 查看参数
beta = model.coef_[0]  # 线性方程的截距
print(alpha, beta)题目二"""



"""model = LinearRegression().fit(dia, price)
twelve = model.predict([[12]])
print("{:.2f}".format(twelve[0]))题目三"""



"""test_data = pd.read_excel("./test.xlsx")
model = LinearRegression().fit(dia, price)
test_dia = [[8], [9], [11], [12], [16]]
test_pre = test_data['价格']
plt.grid(True)  # 用来显示线
plt.xlim((0, 25))  # x轴范围
plt.ylim((0, 25))  # y轴范围
plt.scatter(dia, price, c="red",s=10)  # 绘制散点图
plt.scatter(test_dia, test_pre,s=10)
price_pre = model.predict(dia)
plt.plot(dia, price_pre, c='g')
plt.show()题目四"""
