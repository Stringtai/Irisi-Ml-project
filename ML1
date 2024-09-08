from sys import *
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from scipy import *
from numpy import *
from matplotlib import *
from pandas import *
from sklearn import *
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Grabs dataset from iris.csv and gives them their respected colum
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#Prints the dataset and organizes it
print(dataset.shape) #how many rows
print(dataset.head(20)) #how much rows to print out
print(dataset.describe()) #gives out the statistics for them
print(dataset.groupby('class').size()) #Puts them in their classes e.i length

#makes a new window about dataset
dataset.plot(kind='box', subplots = True, layout = (2,2), sharex = False, sharey = False)
plt.show() #shows the dataset

#gives a scatterplot for data
dataset.hist()
plt.show() #shows data

#Shows the interactions between variables
scatter_matrix(dataset)
plt.show()

#valaid the dataset
array = dataset.values
X = array[:,0:4] #First 4 columns
Y = array[:,4]  #the rest of the columns
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

#checks the Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#Sloves each model
results = []
names = []
for name, model in models:
 kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
 cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
 results.append(cv_results)
 names.append(name)
 print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#Compares all Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

#predictions
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

#Slove predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

