# -*- coding: UTF-8 -*-
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plot

iris = datasets.load_iris()
print iris.data
print iris.target
clf = svm.SVC(gamma=0.001, C=100.)
print clf
clf = svm.SVC(gamma=0.001, C=100.)
digits = datasets.load_digits()
clf.fit(digits.data[:-1], digits.target[:-1])
result=clf.predict(digits.data[0:5])
digits = datasets.load_digits()
print('共'+str(len(digits.data))+'張')
print('預測'+str(digits.target[0:5]))
print('結果'+str(result))
plot.figure(1, figsize=(3, 3))
plot.imshow(digits.images[0], cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()
plot.imshow(digits.images[1], cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()
plot.imshow(digits.images[2], cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()
plot.imshow(digits.images[3], cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()
plot.imshow(digits.images[4], cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()