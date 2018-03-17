from numpy import *
import operator
import csv

'''
    将csv文档转换成向量矩阵，参数为文件名
    前256列为图形矩阵imgMat，后10列为所代表的数字，后10列中第几位是1，将该位置号存入Labels
    返回矩阵和标签
'''
def csv2vec(filename):
    my_matrix = loadtxt(open(filename, "rb"), skiprows = 0)
    a = hsplit(my_matrix,(256,))
    imgMat = a[0]
    resultMat = a[1]
    Labels = []
    for i in range(resultMat.shape[0]):
        for j in range(resultMat.shape[1]):
            if resultMat[i][j] == 1:
                Labels.append(j)
                break
    return imgMat, Labels

'''
    kNN分类器，参数为待分类向量，训练集，标签，k值
    返回距离待分类向量距离最近的k个值中出现次数最多的对应的数字
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #求差
    sqDiffMat = diffMat**2 #差的平方
    sqDistances = sqDiffMat.sum(axis=1) #平方和相加
    distances = sqDistances**0.5 #开方即为欧式距离
    sortedDistIndicies = distances.argsort() #距离升序排序
    classCount = {}
    for i in range(k): #计算距离最小的前k个指分别对应哪个数字
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) #把前k个对应的数字出现次数降序排序
    #print(sortedClassCount)
    return sortedClassCount[0][0] #返回出现最多次数的对应数字

if __name__ == "__main__":
    trainImgMat, trainLabels = csv2vec("../dataSet/semeion_train.csv")
    testImgMat, testLabels = csv2vec("../dataSet/semeion_test.csv")
    error = 0.0
    inputk = input("input the value of k: ")
    k = int(inputk) #输入k值
    f = open("../results/k=%d.txt" % k, "w+") #将结果写入txt文件
    for i in range(testImgMat.shape[0]):
        classfyResult = classify0(testImgMat[i], trainImgMat, trainLabels, k)
        f.write("the classfier came back with: %d, the real answer is: %d\n" % (classfyResult, testLabels[i]))
        if classfyResult != testLabels[i]:
            error += 1.0 #计算错误数
    f.write("the total number of errors is: %d\n" % error)
    f.write("the total error rate is: %f\n" % (error/testImgMat.shape[0]))
    f.close()
    print("results are in the \"results/k=%d.txt\"" % k)
