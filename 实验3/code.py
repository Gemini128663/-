import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('./air_data.csv',encoding='utf-8')
k = 5  # 确定聚类中心数
print(data.head(), '\n')  # 显示数据前五行
kmeans_model = KMeans(n_clusters=k, init='k-means++')
kmeans = kmeans_model.fit(data)

r1 = pd.Series(kmeans_model.labels_).value_counts()  # 统计不同样本的类别数目
print("每个聚类类别的样本数量：")
print(r1, '\n')
print('客户样本类别的标准中心值：')
for i in range(5):
    print(i,  list(kmeans_model.cluster_centers_[i][:5]), '\n')
print('聚类个数及中心点统计:')
print("聚类个数       ZL           ZR                 ZM                        ZF                         ZC")
for i in range(5):
    print(r1[i],  list(kmeans_model.cluster_centers_[i][:5]), '\n')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.title("2005121E04+杜博韬")
plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])  # 规定x轴刻度
plt.yticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # 规定y轴刻度
plt.xlabel("ZL--ZR--ZF--ZM--ZC")
plt.ylabel("Cluster Center Value")
for i in range(5):
    plt.plot([1.0, 2.0, 3.0, 4.0, 5.0], kmeans_model.cluster_centers_[i], marker ='o')
plt.legend(["cluster 0", "cluster 1", "cluster 2", "cluster 3", "cluster 4", ])
plt.show()