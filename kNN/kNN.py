#!/usr/bin/python
# -*- coding:utf8 -*-

from numpy import *
import operator


#创造数据集
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

"""
#将inX扩展到和训练样本集dataSet一样的行数
diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    tile(inX, n):拓展长度
    tile(inX, (m, n):m-拓展个数, n拓展长度

sortedDistIndicies = distances.argsort()#按元素大小升序,将元数对应的索引(index)输出
将算出的距离值元素按升序原则找出对应index. 即找出与测试元素（or测试集）距离最近的几个训练元素的index(下标)
e.g:
    distances:[ 1.00498756  1.          1.          0.9       ]
    sortedDistIndicies:[3 1 2 0]
    

sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
以排出的每组数据的第1(0/1)个元素的大小为准，按降序排列。
e.g:[('A', 2), ('B', 1)]
"""
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  #训练样本行数（矩阵第一维度的长度）
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5 #欧氏距离计算
    #print (distances)
    sortedDistIndicies = distances.argsort()#按元素大小升序,将元数对应的索引(index)输出
    #print (sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  #输出上面相应索引(index)对应的label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #统计label个数
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


"""
file2matrix
函数line.strip()删除字符串中开头、结尾处空白符（包括'\n', '\r',  '\t',  ' ')，
然后使用tab字符\t将上一步得到的整行数据分割成一个元素列表

int(listFormLine[-1]);python中可以使用索引值-1表示列表中的最后一列元素。
此外这里我们必须明确的通知解释器，告诉它列表中存储的元素值为整型，否则python语言会将这些元素当做字符串处理。

strip()使用说明参考: http://www.cnblogs.com/itdyb/p/5046472.html
"""
def file2matrix(filename):
    #得到文件行数
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)

    returnMat = zeros((numberOfLines, 3))   #创建返回的NumPy矩阵

    #解析文件数据到列表
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3] #选取每行的前3个元数存到特征矩阵(的一维)中
        classLabelVector.append(int(listFromLine[-1])) #索引值-1表示列表中的最后一列元素
        index += 1
    return returnMat, classLabelVector

"""
shape用法：
http://jingyan.baidu.com/article/a24b33cd5c90b319fe002b9e.html

"""
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

'''
分类器针对约会网站的测试代码
改进：
1,预测结果不准确的相关详细信息在结尾打印？

'''
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        #print ("the classifier came back with : %d, the real answer is: %d"
               #%(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0

    print ("the total count is: %d" %(errorCount))
    print ("the total error rate is: %f" %(errorCount/float(numTestVecs)))

'''
约会网站预测函数 
'''
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input('percentage of time spent playing video games?'))
    ffMiles = float(raw_input('frequent flier miles earned per year?'))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print ("You will probably like this person: ", resultList[classifierResult-1])

