#!/usr/bin/python
# -*- coding:utf8 -*-
import kNN

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

group, labels = kNN.createDataSet()
print (group)
print (labels)

inX = [0, 1]
testResult = kNN.classify0(inX, group, labels, 3)
print (testResult)


print ('===================Datingdatamat===========================')
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print (datingDataMat[0:10])
print (datingLabels[0:10])

fig = plt.figure() #创建一幅图
#参数111的意思是：将画布分割成1行1列，图像画在从左到右从上到下的第1块
#第十块怎么办，3410是不行的，可以用另一种方式(3,4,10)
#PS:http://blog.163.com/my_it_dream_pwj/blog/static/17841430520112294342649/
ax = fig.add_subplot(111)
#【数字的可视化：python画图之散点图sactter函数详解】
# 参考：http://blog.csdn.net/u013634684/article/details/49646311
#       http://www.cnblogs.com/xiaoyesoso/p/5208079.html
# ax.scatter(x, y, s(大小), c(颜色))
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
           15.0*array(datingLabels), 15.0*array(datingLabels))
#plt.show()

print ('===================归一化===========================')
norMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print (norMat)
print (ranges)
print (minVals)

print ('\n==================分类器针对约会网站的测试代码===================')
#datingTest = kNN.datingClassTest()

datingTest = kNN.classifyPerson()

