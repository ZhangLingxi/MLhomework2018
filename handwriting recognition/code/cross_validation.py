import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

'''
    生成交叉验证集和去掉交叉验证集后的新的训练集
    传入参数为初始训练集文件名
    返回交叉验证集矩阵，交叉验证集标签，训练集矩阵，训练集标签
'''
def generateMat(filename):
    trainImgMat, trainLabels = kNN.csv2vec(filename)
    crossValiMat = zeros((int(trainImgMat.shape[0]/5), int(trainImgMat.shape[1])))
    crossValiLabels = []
    for i in range(crossValiMat.shape[0]): #生成交叉验证集
        crossValiMat[i] = trainImgMat[i*5]
        trainImgMat[i*5,0] = -1 
        crossValiLabels.append(trainLabels[i*5])
    for i in range(trainImgMat.shape[0] - crossValiMat.shape[0]): #去掉交叉验证集，生成新的训练集
        if(trainImgMat[i,0] == -1):
            trainImgMat = delete(trainImgMat, i, 0)
            trainLabels.pop(i)
    return trainImgMat, trainLabels, crossValiMat, crossValiLabels

'''
    根据规定的k值范围，得到交叉验证集的错误率并画出k-errorRate折线图，保存
    传入参数为交叉验证集矩阵，交叉验证集标签，训练集矩阵，训练集标签，k最小值，k最大值
'''
def draw(crossValiMat, crossValiLabels, trainImgMat, trainLabels, mink, maxk):
    figMat = zeros((10,2)) #用于画折线图的矩阵
    k = mink
    f = open("../results/cross_validation.txt" , "w+")
    while ( k >=mink and k <= maxk): #k取值1~numk，分别计算对应的错误率
        error = 0
        for j in range(crossValiMat.shape[0]):
            classfyResult = kNN.classify0(crossValiMat[j], trainImgMat, trainLabels, k)
            if classfyResult != crossValiLabels[j]:
                error += 1.0
        errorRate = error/crossValiMat.shape[0]
        f.write("k = %d, errorRate = %f\n" % (k, errorRate))
        figMat[k][0] = k
        figMat[k][1] = errorRate
        k += 1
    f.close()
    plt.figure()
    plt.plot(figMat[:,0], figMat[:,1]) #画折线图
    plt.savefig("../results/cross_validation.jpg")  #保存图象 
    plt.show()

if __name__ == "__main__":
    crossValiMat, crossValiLabels, trainImgMat, trainLabels = generateMat("../dataSet/semeion_train.csv")
    draw(crossValiMat, crossValiLabels, trainImgMat, trainLabels, 1, 9)
