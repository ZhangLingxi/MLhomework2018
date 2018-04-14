from numpy import *
from math import *
from data import *

N=1000
K=3

'''
    用于计算错误率
    输入参数为矢量数目，实际分类，估计分类
'''
def getErrorRate(N, label, outputLabel):
    errorNum = np.int(np.shape(np.nonzero(np.array(list(outputLabel)) - np.array(list(label))))[1])
    errorRate = float(errorNum) / N
    return errorRate

'''
    计算欧氏距离
'''
def getDistanced(vec1, vec2):
    return sqrt(sum(power(vec1 - vec2,2)))

'''
    用于估计矢量所属类别
'''
def getLabel(data, K, mean_X):
    m, n = shape(data)
    clusterAssment = zeros(m)
    centroids = mat(mean_X)
    for i in range(m):
        minDist = inf
        minIndex = -1
        for j in range(K):
            dist = getDistanced(data[i, :],centroids[j, :])
            if dist < minDist:
                minDist = dist
                minIndex = j
        if clusterAssment[i] != minIndex:
            clusterAssment[i] = minIndex
    return centroids, clusterAssment

centroids1, clusterAssment1 = getLabel(data1, K, mean_X1)
errorRate_X1 = getErrorRate(N, label_data1, clusterAssment1)

centroids2, clusterAssment2 = getLabel(data2, K,mean_X2)
errorRate_X2 = getErrorRate(N, label_data2, clusterAssment2)
print("\n使用欧氏距离分类器，数据集X的错误率为：%f\n" % errorRate_X1)
print("使用欧氏距离分类器，数据集X'的错误率为：%f\n" % errorRate_X2)