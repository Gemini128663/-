import joblib
import numpy as np
from timeit import default_timer as timer
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
tic=timer()
# 加载数据集
data = np.loadtxt("./digits_training.csv", skiprows=1, delimiter=',')

print("训练数据大小为", data.shape[0])
xTrain = data[:, 1:]
yTrain = data[:, 0]


def normalizeData(X):
    """标准化函数"""
    return X - np.mean(X, axis=0)
    # return (X - X.mean())/X.max()


# 数据初始化
print("训练数据特征属性标准化中...")
xTrain = normalizeData(xTrain)
print("降维...")
pca = PCA(n_components=0.65)

xTrain_re = pca.fit_transform(xTrain)  # 先拟合数据，再进行标准化
print('降维之后的主成分个数有', xTrain_re.shape[1])

# 建立模型,拟合mlp模型
print('构建模型中...')
model = MLPClassifier(activation='relu', solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(48, 24))
print('训练模型中...')
model.fit(xTrain_re, yTrain)
# 使用joblib保存模型
print("保存模型中...")
joblib.dump(model, "mlpNN_pca.m")
# 测试集
print("读取测试数据中...")
data2 = np.loadtxt("./digits_testing.csv", skiprows=1, delimiter=',')
print("测试数据大小为", data2.shape[0])
xTest = data2[:, 1:]
yTest = data2[:, 0]
print("测试数据特征属性标准化中...")
xTest = normalizeData(xTest)
# 载入模型
xTest_re = pca.transform(xTest)
print("加载模型中...")
model2 = joblib.load("mlpNN_pca.m")
# 预测模型
print("模型预测中...")
pred = model2.predict(xTest_re)
# 打印错误数据
print("预测错误数据为", (pred != yTest).sum())
# 评价模型
print("预测的准确率为{:.2f}%".format(accuracy_score(yTest, pred) * 100))
toc=timer()
print(toc-tic)
