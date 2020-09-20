import time

import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as kNN

from matplotlib import pyplot as plt


USE_CREDIT = False





# DIABETES

df = pd.read_csv('diabetes.csv') # https://www.openml.org/d/37
numeric = ["preg","plas","pres","skin","insu","mass","pedi","age"]
pos_label = "tested_positive"
df_num = df[numeric]
normalized_df=(df_num-df_num.min())/(df_num.max()-df_num.min()) # https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
df = df.drop(numeric, axis=1)
df = pd.concat([df, normalized_df], axis=1)

data_prefix = "Diabetes Dataset - "


# CREDIT

if USE_CREDIT:

	df = pd.read_csv('credit.csv') # https://www.openml.org/d/31
	pos_label = "good"

	cols_qualitative = ["checking_status","credit_history","purpose","savings_status","employment","personal_status","other_parties","property_magnitude","other_payment_plans","housing","job","own_telephone","foreign_worker"]
	df_1hot = df[cols_qualitative]
	df_1hot = pd.get_dummies(df_1hot).astype('category')
	df_others = df.drop(cols_qualitative, axis=1)
	df = pd.concat([df_others, df_1hot], axis=1)

	cols_quantiative = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 'num_dependents']
	df_num = df[cols_quantiative]
	df_stand =(df_num-df_num.min())/(df_num.max()-df_num.min())
	df_bank_categorical = df.drop(cols_quantiative,axis=1)
	df = pd.concat([df_bank_categorical,df_stand],axis=1)
	df.describe(include='all')

	print(df)

	data_prefix = "Credit Dataset - "








# get train and test

x = df.drop("class", axis=1)
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) # https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas




def trainingSet(x, y, classifier, prefix):
	train = []
	test = []
	sizes = np.arange(0.1, 1, 0.1)
	for i in sizes:
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1 - i)
		classifier.fit(x_train, y_train)
		pred = classifier.predict(x_test)
		train.append(accuracy_score(y_true=y_train, y_pred=classifier.predict(x_train)))
		test.append(accuracy_score(y_true=y_test, y_pred=pred))

	plt.plot(sizes, test, 'o-', color='g', label='Test Accuracy')
	plt.plot(sizes, train, 'o-', color = 'r', label='Train Accuracy')
	plt.title(data_prefix + prefix + " Training Size vs Accuracy")
	plt.xlabel('Training Size')
	plt.ylabel('Accuracy')

	plt.legend(loc='best')
	plt.show()

	min_diff = 1
	best_index = 0
	for i in range(0, len(train)):
		if abs(train[i] - test[i]) < min_diff:
			min_diff = abs(train[i] - test[i])
			best_index = i

	best_test_size = 1 - sizes[best_index]
	return best_test_size


def plot_hyper(test, train, data, prefix):
	plt.plot(data, test, 'o-', color='g', label='Test Score')
	plt.plot(data, train, 'o-', color = 'r', label='Train Score')
	plt.title(data_prefix + prefix + " Hyperparameter")
	plt.xlabel(prefix)
	plt.ylabel('Score')

	plt.legend(loc='best')
	plt.show()


def print_accuracy(best_test_size, classifier):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = best_test_size)
	classifier.fit(x_train, y_train)
	print('Train Accuracy: ', accuracy_score(y_true=y_train, y_pred=classifier.predict(x_train)))
	print('Test Accuracy: ', accuracy_score(y_true=y_test, y_pred=classifier.predict(x_test)))



# Decision tree
# https://heartbeat.fritz.ai/decision-tree-classification-in-python-with-scikit-learn-245502ada8aa
# https://towardsdatascience.com/decision-tree-build-prune-and-visualize-it-using-python-12ceee9af752
	
start = time.time()

test = []
train = []
max_depth = list(range(1, 20))
for i in max_depth:         
		clf = DecisionTreeClassifier(max_depth=i, random_state=100, min_samples_leaf=1, criterion='entropy')
		clf.fit(x_train, y_train)
		y_pred_test = clf.predict(x_test)
		y_pred_train = clf.predict(x_train)
		test.append(f1_score(y_test, y_pred_test, pos_label=pos_label))
		train.append(f1_score(y_train, y_pred_train, pos_label=pos_label))

plot_hyper(test, train, max_depth, "Max Depth")


param_grid = {
	'criterion': ['gini', 'entropy'],
	'class_weight': ['balanced', None],
	'min_samples_leaf': np.arange(1, 20),
	'max_depth': np.arange(1,20)
}

tree = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=10)
tree.fit(x_train, y_train)
print("Best params: ", tree.best_params_)

dtree = DecisionTreeClassifier(max_depth=tree.best_params_['max_depth'], min_samples_leaf=tree.best_params_['min_samples_leaf'], random_state=100, criterion=tree.best_params_['criterion'], class_weight=tree.best_params_['class_weight'])
best_test_size = trainingSet(x, y, dtree, "Decision Tree")
end = time.time()

print("Training time (s): ", end - start)
print ("Best test size: ", best_test_size)

print_accuracy(best_test_size, dtree)






# Neural Networks

start = time.time()

test = []
train = []
data = np.linspace(1,150,30).astype('int')
for i in data:         
		clf = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic', 
							learning_rate_init=0.05, random_state=100)
		clf.fit(x_train, y_train)
		y_pred_test = clf.predict(x_test)
		y_pred_train = clf.predict(x_train)
		test.append(f1_score(y_test, y_pred_test, pos_label=pos_label))
		train.append(f1_score(y_train, y_pred_train, pos_label=pos_label))
  
plot_hyper(test, train, data, "Hidden Layers")

h_units = np.arange(10, 110, 10)
learning_rates = [0.01, 0.02, 0.05, .1]
param_grid = {'hidden_layer_sizes': h_units, 'learning_rate_init': learning_rates}

tree = GridSearchCV(estimator = MLPClassifier(solver='adam',activation='logistic',random_state=100),
				   param_grid=param_grid, cv=10)
tree.fit(x_train, y_train)
print("Best params: ", tree.best_params_)

dtree = MLPClassifier(hidden_layer_sizes=(tree.best_params_['hidden_layer_sizes'],), solver='adam', activation='logistic', 
							   learning_rate_init=tree.best_params_['learning_rate_init'], random_state=100)

best_test_size = trainingSet(x, y, dtree, "Neural Networks")
end = time.time()
print("Training time (s): ", end - start)
print ("Best test size: ", best_test_size)

print_accuracy(best_test_size, dtree)







# Boosting

start = time.time()


test = []
train = []
data = np.linspace(1,250,40).astype('int')
for i in data:         
		clf = GradientBoostingClassifier(n_estimators=i, max_depth=3, 
										 min_samples_leaf=25, random_state=100,)
		clf.fit(x_train, y_train)
		y_pred_test = clf.predict(x_test)
		y_pred_train = clf.predict(x_train)
		test.append(f1_score(y_test, y_pred_test, pos_label=pos_label))
		train.append(f1_score(y_train, y_pred_train, pos_label=pos_label))

plot_hyper(test, train, data, "Number of Estimators")

param_grid = {
	'min_samples_leaf': np.linspace(1,30,5).round().astype('int'),
	'max_depth': [1,2,3],
	'n_estimators': [10, 30, 50, 70, 100],
	'learning_rate': [0.01, 0.02, 0.05, .1]
}

tree = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid=param_grid, cv=10)
tree.fit(x_train, y_train)
print("Per Hyperparameter tuning, best parameters are:")
print("Best params: ", tree.best_params_)

dtree = GradientBoostingClassifier(max_depth=tree.best_params_['max_depth'], min_samples_leaf=tree.best_params_['min_samples_leaf'], 
											  n_estimators=tree.best_params_['n_estimators'], learning_rate=tree.best_params_['learning_rate'], random_state=100)

best_test_size = trainingSet(x, y, dtree, "Boosting")
end = time.time()
print("Training time (s): ", end - start)
print ("Best test size: ", best_test_size)

print_accuracy(best_test_size, dtree)






# Support Vector Machines

start = time.time()

test = []
train = []
kernels = ['poly','poly','poly','poly','poly','poly', 'linear', 'rbf','sigmoid']
for i in range(2, len(kernels) + 2):          
	lol = SVC(kernel=kernels[i-2], degree=i, random_state=100)
	lol.fit(x_train, y_train)
	y_pred_test = lol.predict(x_test)
	y_pred_train = lol.predict(x_train)
	test.append(f1_score(y_test, y_pred_test, pos_label=pos_label))
	train.append(f1_score(y_train, y_pred_train, pos_label=pos_label))
			
data = ['poly2','poly3','poly4','poly5','poly6','poly7', 'linear', 'rbf','sigmoid']
plot_hyper(test, train, data, "Kernel")


Cs = [1e-4, 1e-3, 1e-2, 1e01, 1]
gammas = ["scale", "auto"]
param_grid = {'C': Cs, 'gamma': gammas }

tree = GridSearchCV(estimator = SVC(random_state=100, kernel='rbf'), 
				   param_grid=param_grid, cv=10)
tree.fit(x_train, y_train)
print("Best params: ", tree.best_params_)

dtree = SVC(C=tree.best_params_['C'], gamma=tree.best_params_['gamma'], kernel='rbf', random_state=100)

best_test_size = trainingSet(x, y, dtree, "Support Vector Machines")
end = time.time()
print("Training time (s): ", end - start)
print ("Best test size: ", best_test_size)

print_accuracy(best_test_size, dtree)


# KNN

start = time.time()
test = []
train = []
data = np.linspace(1,len(x_train),25).astype('int')
for i in data:
	clf = kNN(n_neighbors=i,n_jobs=-1)
	clf.fit(x_train,y_train)
	y_pred_test = clf.predict(x_test)
	y_pred_train = clf.predict(x_train)
	test.append(f1_score(y_test, y_pred_test, pos_label=pos_label))
	train.append(f1_score(y_train, y_pred_train, pos_label=pos_label))
	
plot_hyper(test, train, data, "Number of Neighbors")

dtree = kNN(n_neighbors=20, n_jobs=-1)
best_test_size = trainingSet(x, y, dtree, "KNN")
end = time.time()
print("Training time (s): ", end - start)
print ("Best test size: ", best_test_size)

print_accuracy(best_test_size, dtree)
