from numpy import *
import csv
import pandas as pd
from math import sqrt

'''
    将csv文档转换成向量矩阵，参数为文件名，最后一列单独存为类别向量
    返回矩阵和类别向量
'''
def csv2vec(filename):
    df = pd.read_csv(filename, skiprows = 0, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9))
    df.values
    mat = df.as_matrix(columns=None)
    label = []
    for i in range(len(mat)):
        label.append(mat[i][-1])
    return mat, label

train_set, train_label = csv2vec('../data/train.csv')
plabel1 = (sum(train_label) + 1) / (len(train_set) + 2)
plabel0 = 1 - plabel1

'''
    前6个特征的概率计算
'''
def pFeature(train_set):
    # p0Vec三行六列，对应分类为0时，六个特征在取1，2，3三个不同值时候的概率
    # 进行拉普拉斯平滑处理
    p0Vec = ones([3,6]) 
    # 进行拉普拉斯平滑处理
    # p1Vec三行六列，对应分类为1时，六个特征在取1，2，3三个不同值时候的概率
    p1Vec = ones([3,6]) 
    count0 = 3 # 拉普拉斯平滑
    count1 = 3 # 拉普拉斯平滑
    for item in train_set:
        if item[-1] == 0:
            count0 += 1
            for j in range(6):
                if item[j] == 1:
                    p0Vec[0][j] +=1
                if item[j] == 2:
                    p0Vec[1][j] += 1
                if item[j] == 3:
                    p0Vec[2][j] += 1    
        if item[-1] == 1:
            count1 += 1
            for j in range(6):
                if item[j] == 1:
                    p1Vec[0][j] +=1
                if item[j] == 2:
                    p1Vec[1][j] += 1
                if item[j] == 3:
                    p1Vec[2][j] += 1  
    p0Vec = log(p0Vec / count0)
    p1Vec = log(p1Vec / count1)
   # print(p0Vec,p1Vec)
    return p0Vec, p1Vec

# 第7个特征的均值与标准差
getx = lambda dataset, y, index: [x[index] for x in dataset if x[-1] == y]
mean7_1 = mean(getx(train_set, 1.0, 6))
std7_1 = std(getx(train_set, 1.0, 6))
mean7_0 = mean(getx(train_set, 0.0, 6))
std7_0 = std(getx(train_set, 0.0, 6))

# 第8个特征的均值与标准差
mean8_1 = mean(getx(train_set, 1.0, 7))
std8_1 = std(getx(train_set, 1.0, 7))
mean8_0 = mean(getx(train_set, 0.0, 7))
std8_0 = std(getx(train_set, 0.0, 7))

# 7 8特征概率的计算
def p_continuity_feature(dataset, value7, value8, pos=True):
    p7_1 = 0.0; p8_1 = 0.0; p7_0 = 0.0; p8_0 = 0.0
    if pos:
        p7_1 = log((1 / (sqrt(2 * pi) * std7_1)) * exp(-((value7 - mean7_1) ** 2 / (2 * (std7_1 ** 2)))))
        p8_1 = log((1 / (sqrt(2 * pi) * std8_1)) * exp(-((value8 - mean8_1) ** 2 / (2 * (std8_1 ** 2)))))
    else:
        p7_0 = log((1 / (sqrt(2 * pi) * std7_0)) * exp(-((value7 - mean7_0) ** 2 / (2 * (std7_0 ** 2)))))
        p8_0 = log((1 / (sqrt(2 * pi) * std8_0)) * exp(-((value8 - mean8_0) ** 2 / (2 * (std8_0 ** 2)))))
    #print(p7_1, p8_1) if pos else print(p7_0, p8_0)
    return (p7_1, p8_1) if pos else (p7_0, p8_0)

'''
    朴素贝叶斯分类器，传入参数为待检验的特征向量
    返回依照朴素贝叶斯算法预测的分类结果
'''
def classify(dataVec):
    p0Vec, p1Vec = pFeature(train_set)
    p1 = 1.0; p0 = 1.0
    for j in range(6):
        p1 += p1Vec[int(dataVec[j])-1][j]
        p0 += p0Vec[int(dataVec[j])-1][j]
    p1 += (lambda x: x[0]*x[1])(p_continuity_feature(train_set, dataVec[6], dataVec[7]))
    p0 += (lambda x: x[0]*x[1])(p_continuity_feature(train_set, dataVec[6], dataVec[7], pos=False))
    p1 += log(plabel1)
    p0 += log(plabel0)
   # print(p1)
    #print(p0)
    if p1 > p0:
        return 1
    else:
        return 0

test_set, test_label = csv2vec("../data/test.csv")
correct = 0.0
error = 0.0
f = open("../result/result.txt", "w+") #将结果写入txt文件   
for i in range(len(test_set)):
    result = classify(test_set[i])
    f.write("the classifier came back with: %d, the real label is: %d\n" % (result, test_label[i]))
    if result == test_label[i]:
        correct += 1
    else:
        error += 1
precision = correct / (correct + error)
f.write("the number of errors is: %d\n" % error)
f.write("the precision is: %f\n" % precision)
f.close()
print("results are in the \"result/result.txt\"")