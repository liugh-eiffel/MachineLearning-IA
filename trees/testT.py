#!/usr/bin/python
# -*- coding:utf8 -*-

import trees

myDat, labels = trees.createDataSet()
print (myDat)
calc = trees.calcShannonEnt(myDat)
print (calc)

print ('\n--------------------------------------')
testSd = trees.splitDataSet(myDat, 1, 1)
print (testSd)

print ('\n--------------------------------------')
testCb = trees.chooseBestFeatureToSplit(myDat)
print (testCb)
