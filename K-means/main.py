import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs

import stft
from stft import loaddata
from sklearn import metrics
import matplotlib.pyplot as plt

def loadDataCsv(path):
    f = open(path,"r")
    print("ok")
    res = pd.read_csv(f, delimiter=",",header=None, skiprows=0)
    data = res.to_numpy()
    datalabel = data.copy()
    # print(data)
    n = len(data[:])
    print(n)
    label = data[:, -4:]
    data = np.delete(data,(-1), axis=1)
    print("data.shape",data.shape)
    print("label.shape",label.shape)
    return data,label,datalabel

# fname = "data-tt.csv"
# data1, label1, datalabel1 = loadDataCsv(fname)
# demo_stft.csv
res = pd.read_csv('./result/demo_stft.csv', delimiter=",", header=None, skiprows=0)
data1 = res.to_numpy()
print(data1.shape)
x = data1
#
# # x,y = make_blobs(n_samples=1000,n_features=4,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.4],random_state=10)
# print(x.shape,y.shape)
k_means = KMeans(n_clusters=4, random_state=42)
#
plt.plot(x)
plt.show()
# print(label1)
k_means.fit(x)
#
y_predict = k_means.predict(x)
plt.plot(y_predict)
plt.show()
# plt.scatter(x[:,0],x[:,1],c=y_predict)
# plt.show()
print(y_predict)
# print(k_means.predict((x[:30,:])))
# # print(metrics.calinski_harabaz_score(x,y_predict))
# print(metrics.calinski_harabasz_score(x, y_predict))
# print(k_means.cluster_centers_)
# print(k_means.inertia_)
# print(metrics.silhouette_score(x,y_predict))
stft.loaddata('data-nolabel.csv')
