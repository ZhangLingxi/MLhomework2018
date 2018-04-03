from numpy import *
import csv
import operator
import pandas as pd
from math import log

'''
    将csv文件转换成矩阵mat和向量label
    矩阵包括特征值和分类
    向量为特征，即['色泽', '根蒂','敲声','纹理']
'''
def csv2vec(filename):
    df = pd.read_csv(filename, skiprows = 0, usecols=(1, 2, 3, 4, 5), encoding = 'gbk')
    df.values
    mat = df.as_matrix(columns = None)
    df1 = pd.read_csv(filename, nrows = 0, usecols=(1, 2, 3, 4), encoding = 'gbk')
    label = list(df1)
    #print(mat)
    #print(label)
    return mat, label

'''
    将特征值为中文的矩阵转换成用数字表示，输入参数为待转换矩阵
    每个特征的特征值取值范围为0~特征值取值数-1
    并将特征值及转换后对应的数字存入字典featDict
'''
def convertMat(matrix):
    featDict = {}
    for j in range(len(matrix[0]) - 1):
        num = 0
        for i in range(len(matrix)):
            if matrix[i][j] in featDict:
                continue
            else:
                featDict[matrix[i][j]] = num
                num += 1
    matList = [[0 for col in range(len(matrix[0]))] for row in range(len(matrix))]  
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if j < len(matrix[0]) - 1:
                matList[i][j] = int(featDict[matrix[i][j]])
            else:
                matList[i][j] = matrix[i][j]
    #print(featDict)
    return featDict, matList

'''
    计算香农熵
    输入数据集，返回香农熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

'''
    按照离散变量划分数据集，取出该特征取值为value的所有样本
    输入参数为：待划分的数据集，划分数据集的特征，需要返回的特征的值
    返回划分后的数据集
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:]) #将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
    选择最好的数据集划分方式
    返回信息增益最大的(最优)特征的索引值
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0 #信息增益
    bestFeature = -1 #最优特征的索引值
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''
    统计classList中出现此处最多的元素(类标签)
    出现此处最多的元素(类标签)
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

'''
    递归构建决策树
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 2: #遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

'''
    根据决策树预测测试集结果
'''
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    #print(featIndex)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    myDatMat, myLabel = csv2vec("../dataSet/Watermelon-train1.csv")
    labels = myLabel[:]
    featDict, myDat = convertMat(myDatMat)
    myTree = createTree(myDat, myLabel)
    print("============================================================================================================")
    print("决策树如下:\n")
    print(myTree)
    print("============================================================================================================")
    print("测试集预测结果如下:\n")
    testDatMat, testLabel = csv2vec("../dataSet/Watermelon-test1.csv")
    testDat = [[0 for col in range(len(testDatMat[0]) - 1)] for row in range(len(testDatMat))]  
    for i in range(len(testDatMat)):
        for j in range(len(testDatMat[0]) - 1):
            testDat[i][j] = int(featDict[testDatMat[i][j]])
    correct = 0.0
    error = 0.0 
    for i in range(len(testDat)):
        result = classify(myTree, labels, testDat[i])
        print("编号为%d的预测分类是: %s, 实际分类是: %s\n" % (i+1, result, testDatMat[i][-1]))
        if result == testDatMat[i][-1]:
            correct += 1
        else:
            error += 1
    print("预测错误有: %d个\n" % error)
    precision = correct / (correct + error)
    print("预测准确率为: %f\n" % precision)
