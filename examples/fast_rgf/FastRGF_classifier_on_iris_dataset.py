import time

import numpy
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from rgf.sklearn import RGFClassifier, FastRGFClassifier

iris = datasets.load_iris()
iris.data = numpy.vstack((iris.data, iris.data, iris.data, iris.data, iris.data))
iris.data = numpy.vstack((iris.data, iris.data, iris.data, iris.data, iris.data))
iris.data = numpy.vstack((iris.data, iris.data, iris.data, iris.data, iris.data))
iris.data = numpy.vstack((iris.data, iris.data, iris.data, iris.data, iris.data))
print('data shape ' + str(iris.data.shape))
iris.target = numpy.hstack((iris.target, iris.target, iris.target, iris.target, iris.target))
iris.target = numpy.hstack((iris.target, iris.target, iris.target, iris.target, iris.target))
iris.target = numpy.hstack((iris.target, iris.target, iris.target, iris.target, iris.target))
iris.target = numpy.hstack((iris.target, iris.target, iris.target, iris.target, iris.target))

start = time.time()
clf = RGFClassifier()
clf.fit(iris.data, iris.target)
end = time.time()
print("RGF fit: {} sec".format(end - start))
start = end
score = clf.score(iris.data, iris.target)
end = time.time()
print("RGF predict: {} sec".format(end - start))
print("score: {}".format(score))

start = time.time()
clf = FastRGFClassifier(n_estimators=1000)
clf.fit(iris.data, iris.target)
end = time.time()
print("FastRGF fit: {} sec".format(end - start))
start = end
score = clf.score(iris.data, iris.target)
end = time.time()
print("FastRGF predict: {} sec".format(end - start))
print("score: {}".format(score))

start = time.time()
clf = GradientBoostingClassifier()
clf.fit(iris.data, iris.target)
score = clf.score(iris.data, iris.target)
end = time.time()
print("Gradient Boosting: {} sec".format(end - start))
print("score: {}".format(score))
