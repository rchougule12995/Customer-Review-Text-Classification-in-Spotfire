import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

dataset1 = dataset
X_features = vectorized_data

original_ds = dataset.copy()

y = dataset1['rating_type'].values

X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size = 0.20, random_state = 25)

# Function to train the Naive Bayes Classifier Model
def nBayesModel_train():
    nbModel = MultinomialNB()
    nbModel.fit(X_train, y_train)
    return nbModel

# Function to evaluate the Naive Bayes Classifier Model
def evaluate_NB(model, test_features, test_labels):
	predictions = model.predict(test_features)
	precision, recall, fscore, support = score(test_labels, predictions, pos_label='positive', average='binary')
	accuracy = model.score(test_features, test_labels)
	tn, fp, fn, tp = confusion_matrix(test_labels,predictions).ravel()
	print("------------ Naive Bayes Classifier Performance Metrics ------------")
	print("F-score: {} ".format(round(fscore,3)*100))
	print("Precision: {} ".format(round(precision,3)*100))
	print("Recall: {} ".format(round(recall,3)*100))
	print("Accuracy: {} ".format(round(accuracy,3)*100))
	nbMetrics = [precision, recall, fscore, accuracy, tn, fp, fn, tp]
	print("Wrote to nbmetrics")
	return nbMetrics

nb_Model = nBayesModel_train()

nbMetrics = evaluate_NB(nb_Model, X_test, y_test)

#_nbMetrics = pd.DataFrame(nbMetrics)

#Setting a range of parameters to be used in the randomforest classifier to find the best suited parameter for the model
# Number of trees in random forest
n_estimators = [int(x) for x in range(200, 4000, 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', "None"]
# Maximum number of levels in tree
max_depth = [int(x) for x in range(100, 500, 10)]
max_depth.append(None)

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth}

# Using the random grid to search for best hyperparameters in the random forest classifier
rf = RandomForestClassifier()
# Random search of parameters, using default 5 fold cross validation, 
# search across different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

# Function to train the Random Forest Model
def rforestModel_train():
    n_estimators = rf_random.best_params_["n_estimators"]
    max_depth = rf_random.best_params_["max_depth"]
    max_features = rf_random.best_params_["max_features"]
    rf_Model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, 
                                    max_features = max_features)
    rf_Model.fit(X_train, y_train)
    return rf_Model

# Function to evaluate the random forest model
def evaluate_RF(model, test_features, test_labels, n_est, depth):
    predictions = model.predict(test_features)
    precision, recall, fscore, support = score(test_labels, predictions, pos_label='positive', average='binary')
    accuracy = model.score(test_features, test_labels)
    tn, fp, fn, tp = confusion_matrix(test_labels,predictions).ravel()
    print("------------ Random Forest Classifier Performance Metrics ------------")
    print("F-score: {} ".format(round(fscore,3)*100))
    print("Precision: {} ".format(round(precision,3)*100))
    print("Recall: {} ".format(round(recall,3)*100))
    print("Accuracy: {} ".format(round(accuracy,3)*100))
    rfMetrics = [precision, recall, fscore, accuracy, tn, fp, fn, tp]
    return rfMetrics

rf_Model = rforestModel_train()
rfMetrics = evaluate_RF(rf_Model, X_test, y_test, n_estimators, max_depth)

# Function to train the Logistic Regression Model
def logRegModel_train():
    logReg = LogisticRegression(solver='liblinear', C=5, penalty='l2',max_iter=3000)
    logReg.fit(X_train,y_train)
    return logReg
	
# Function to evaluate the Logistic Regression Model
def evaluate_LogReg(model, test_features, test_labels):
    predictions = model.predict(test_features)
    precision, recall, fscore, support = score(test_labels, predictions, pos_label='positive', average='binary')
    accuracy = model.score(test_features, test_labels)
    tn, fp, fn, tp = confusion_matrix(test_labels,predictions).ravel()
    print("------------ Logistic Regression Classifier Performance Metrics ------------")
    print("F-score: {} ".format(round(fscore,3)*100))
    print("Precision: {} ".format(round(precision,3)*100))
    print("Recall: {} ".format(round(recall,3)*100))
    print("Accuracy: {} ".format(round(accuracy,3)*100))
    logRegMetrics = [precision, recall, fscore, accuracy, tn, fp, fn, tp]
    return logRegMetrics

logReg_Model = logRegModel_train()

logRegMetrics_1 = evaluate_LogReg(logReg_Model, X_test, y_test)

classifierEvaluation = pd.DataFrame({
    'Naive Bayes': nbMetrics,
    'Random Forest': rfMetrics,
    'Logistic Regression': logRegMetrics_1
}, index=['precision', 'recall', 'fscore', 'accuracy','true negatives','false positives','false negatives','true positives']
).T
classifierEvaluation["precision"] = round(classifierEvaluation["precision"]*100,2)
classifierEvaluation["recall"] = round(classifierEvaluation["recall"]*100,2)
classifierEvaluation["fscore"] = round(classifierEvaluation["fscore"]*100,2)
classifierEvaluation["accuracy"] = round(classifierEvaluation["accuracy"]*100,2)

classifierEvaluation["Classifer Model"] = classifierEvaluation.index

classifierEvaluation
