from sklearn.datasets import make_classification
import numpy as np
import csv
import pandas as pan
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

nb_samples = 5000
nb_features = 100

X, Y=make_classification(n_samples=nb_samples,
				 n_features=nb_features,
				 n_informative=nb_features-15,
				 n_redundant=0,
				 n_repeated=0,
				 n_classes=2,
				 n_clusters_per_class=2
				 )

# Set random seed (for reproducibility)
np.random.seed(1000)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=1000)

# Perform a logistic regression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
print('Logistic Regression score: {}'.format(lr.score(X_test, Y_test)))

# Set the y=0 labels to -1
Y_train[Y_train==0] = -1
Y_test[Y_test==0] = -1

# Train a Kernel Support Vector Machine
KernelSVM = SVC(C=1.0, kernel='poly', degree=2)
KernelSVM.fit(X_train, Y_train)
print('Kernel SVM score: {}'.format(KernelSVM.score(X_test, Y_test)))

# Train a Multi Layer Perceptron Classifer
MLP = MLPClassifier(hidden_layer_sizes=(200,),
					activation="relu",
					solver="adam",
					batch_size=50)
MLP.fit(X_train, Y_train)
print('MLP score: {}'.format(MLP.score(X_test, Y_test)))


# Passive Aggressive Classifier 

C = 0.01
w = np.zeros((nb_features, 1))

# Implement a Passive Aggressive Classification
for i in range(X_train.shape[0]):
    xi = X_train[i].reshape((nb_features, 1))
    
    loss = max(0, 1 - (Y_train[i] * np.dot(w.T, xi)))
    tau = loss / (np.power(np.linalg.norm(xi, ord=2), 2) + (1 / (2*C)))
    
    coeff = tau * Y_train[i]
    w += coeff * xi
    
# Compute accuracy
Y_pred = np.sign(np.dot(w.T, X_test.T))
c = np.count_nonzero(Y_pred - Y_test)
print('PA accuracy: {}'.format(1 - float(c) / X_test.shape[0]))


# Train an Stochastic Gradient Descent Classifer

poly = PolynomialFeatures(degree=2)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)

SGDC = SGDClassifier(alpha=0.01, loss='hinge', penalty='l2', fit_intercept = True, tol= 1e-3, n_jobs=-1)
SGDC.fit(X_train, Y_train)
print('SGDClassifier score: {}'.format(SGDC.score(X_test, Y_test)))

#  Passive Aggressive Classifier 

PA = PassiveAggressiveClassifier(C=0.01, loss='squared_hinge', n_jobs=-1)
PA.fit(X_train, Y_train)
print('PA score: {}'.format(PA.score(X_test, Y_test)))

