from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import joblib

print('载入训练数据中......')
data = np.loadtxt("digits_training.csv", skiprows=1, delimiter=',')
print("训练数据有：{:}行".format(data.shape[0]))
target_train = data[:, 0]
data_train = data[:, 1:]


def normalizedata(x):
    return (x - x.mean()) / x.max()


print('标准化训练数据中......')
data_train = normalizedata(data_train)
print("构建并训练模型中......")
model = SVC(decision_function_shape='ovo').fit(data_train, target_train)
print('保存模型中......')
joblib.dump(model, 'svm_classifier_model1.m')

print('载入测试数据中......')
data1 = np.loadtxt("digits_testing.csv", skiprows=1, delimiter=',')
print("训练数据有：{:}行".format(data1.shape[0]))

test_data = data1[:, 1:]
test_target = data1[:, 0]
print('测试数据标准化中......')
test_data = normalizedata(test_data)
print('加载模型中......')
model1 = joblib.load('svm_classifier_model1.m')
pred = model1.predict(test_data)

print("预测错误数据为", (pred != test_target).sum())
print("预测的准确率为{:.2f}%".format(accuracy_score(test_target, pred) * 100))

print('模型内建准确率为{:.2f}%'.format((1-(((pred != test_target).sum())/data1.shape[0]))*100))