import numpy as np
import math
from data import *

K=3

PrioPro_data1 = np.array((n1_data1, n2_data1, n3_data1)) / 1000.
PrioPro_data2 = np.array((n1_data2, n2_data2, n3_data2)) / 1000.

'''
    用于计算最大后验概率
    输入参数为类别数，数据集，协方差矩阵，均值，先验概率
'''
def getPosterPro(K, data, sigma, mu, PrioPro):
    m, n = np.shape(data)
    Px_w = np.mat(np.zeros((m, K)))
    for i in range(K):
        coef = (2 * math.pi) ** (-n / 2.) * (np.linalg.det(sigma[i]) ** (-0.5))
        temp = np.multiply((data - mu[i]) * np.mat(sigma[i]).I, data - mu[i])
        Xshift = np.sum(temp, axis=1)
        Px_w[:, i]= coef * np.exp(Xshift * -0.5)  #矩阵与常数相乘
    PosterPro = np.mat(np.zeros((m, K)))
    for i in range(K):
        PosterPro[:, i] = PrioPro[i] * Px_w[:, i]
    return PosterPro

'''
    用于根据后验概率估计矢量所属类别
'''
def getLikelihoodLabel(PosterPro):
    outputLabel = np.argmax(PosterPro, axis = 1)
    outputLabel = map(int, np.array(outputLabel.flatten())[0])
    return outputLabel

'''
    用于计算错误率
    输入参数为矢量数目，实际分类，估计分类
'''
def getErrorRate(N, label, outputLabel):
    errorNum = np.int(np.shape(np.nonzero(np.array(list(outputLabel)) - np.array(list(label))))[1])
    errorRate = float(errorNum) / N
    return errorRate

PosterPro_data1 = getPosterPro(K, data1, sigma_X1, mean_X1, PrioPro_data1)
likelihoodLabel = getLikelihoodLabel(PosterPro_data1)
errorRate_data1_MaxPost = getErrorRate(N, label_data1, likelihoodLabel)

PosterPro_data2 = getPosterPro(K, data2, sigma_X2, mean_X2, PrioPro_data2)
likelihoodLabel_data2 = getLikelihoodLabel(PosterPro_data2)
errorRate_data2_MaxPost = getErrorRate(N, label_data2, likelihoodLabel_data2)

print("\n使用贝叶斯分类器，数据集X的错误率为：%f\n" % errorRate_data1_MaxPost)
print("使用贝叶斯分类器，数据集X'的错误率为：%f\n" % errorRate_data2_MaxPost)

