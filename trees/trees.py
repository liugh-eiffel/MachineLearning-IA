#!/usr/bin/python
# -*- coding:utf8 -*-

import operator
from math import log

'''
calcShannonEnt
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
        #print ('key:%s' % key)
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

'''
createDataSet
'''
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

'''
splitDataSet

a = [1, 2, 3]
b = [4, 5, 6]

a.append(b) => [1, 2, 3, [4, 5, 6]]
a.extend(b) => [1, 2, 3, 4, 5, 6]

以每一维第axis列的数据用于筛选，value为筛选标准
如果符合，每次返回除axis外，每维“切割”后的数据

针对矩阵x操作
x[m:n]    #多个元素，左闭右开，默认步长值是1

'''
def splitDataSet ( dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
chooseBestFeatureToSplit
信息增益：熵的减少或者是数据无序度的减少
获得信息增益最高的特征就是最好的选择。
'''
def chooseBestFeatureToSplit (dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if ( infoGain > bestInfoGain ):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''
majorityCnt :
“多数表决”
创建键值为classList中唯一值的数据字典， 字典对象存储了classList中每个类标签出现的频率，
最后利用operator操作键值排序字典，并返回出现次数最多的分类名称。
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] = 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

'''
createTree
创建树的函数
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):#计算类别元素是否唯一
        print ('==0\n')
        return classList[0]         #类别完全相同则停止继续划分
    if len(dataSet[0]) == 1:        #遍历完所有特征时返回出现次数最多的类别
        print ('==1')
        return majorityCnt(classList) #此时dataSet[0]==example[-1]
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
                                                               bestFeat,
                                                               value), subLabels)
    return myTree