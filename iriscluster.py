#importing the iris datasets
from sklearn import datasets
iris = datasets.load_iris()

#plot petal length and sepal width of the three types of flowers
import matplotlib.pyplot as plt
plt.scatter(iris.data[:,1], iris.data[:,2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

#plotting all the 4 characteristics of setosa which is contained in lines 0--50
plt.scatter(iris.data[0:50,0],iris.data[0:50,1], c=iris.target[0:50])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
plt.scatter(iris.data[0:50,2], iris.data[0:50,3], c=iris.target[0:50])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

#plotting all the 4 characteristics of versicolor which is contained in lines 51--100
plt.scatter(iris.data[51:100,0],iris.data[51:100,1], c=iris.target[51:100])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
plt.scatter(iris.data[51:100,2], iris.data[51:100,3], c=iris.target[51:100])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()


#plotting all the 4 characteristics of virginica which is contained in lines 101--150
plt.scatter(iris.data[101:150,0],iris.data[101:150,1], c=iris.target[101:150])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
plt.scatter(iris.data[101:150,2], iris.data[101:150,3], c=iris.target[101:150])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

#plotting sepal length, sepal width of all the three
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

#plotting petal length, petal width of all the three
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

