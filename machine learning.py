from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import binarize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# features info : https://goo.gl/p8ocBn
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# initialization of the Dataset
data = pd.read_csv('Downloads/diabetes.csv')

data.head()

print(data.describe())
X = data.iloc[:, 0:8]
y = data.iloc[:, 8]

sns.heatmap(X.corr(), annot = True)

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    X[column] = X[column].replace(0, np.NaN)
    mean = int(X[column].mean(skipna=True))
    X[column] = X[column].replace(np.NaN, mean)
    
# Var[X] = p(1-p)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_filtered = sel.fit_transform(X)

print(X.head(1))
print(X_filtered[0])
#DiabetesPedigreeFunction was dropped
X = X.drop('DiabetesPedigreeFunction', axis=1)

top_4_features = SelectKBest(score_func=chi2, k=4)
X_top_4_features = top_4_features.fit_transform(X, y)
print(X.head())
print(X_top_4_features)
X = X.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], axis=1)

## SVM classifer
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = SVC(random_state=0, kernel='rbf')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print (cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

### gradient boosting classifier

%%time
# Magic keyword time who display the cpu usage
# GridSearch procedure = cpu demanding
gradient_boost_eval = GradientBoostingClassifier(random_state=0)

params = {
    'learning_rate': [0.05, 0.1, 0.5],
    'max_features': [0.5, 1],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(gradient_boost_eval, params, cv=10, scoring='roc_auc')
grid_search.fit(X, y)

print(grid_search.best_params_)
print(grid_search.best_score_)

# GBC belong to the Random Tree Classifier family
# a complex model who perform well on small Dataset
# We set our hyper-params with the best params providing by GridSearch
gradient_boost = GradientBoostingClassifier(
    learning_rate=0.05, max_depth=3, max_features=0.5, random_state=0)

#Cross Validation handle the .fit and .predict method
cross_val_score(gradient_boost, X, y, cv=10, scoring='roc_auc').mean()

gradient_boost_imp = GradientBoostingClassifier(
    learning_rate=0.05, max_depth=3, max_features=0.5, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

gradient_boost_imp.fit(X_train, y_train)

# Storing the predicted probabilities of classification
y_pred_prob = gradient_boost_imp.predict_proba(X_test)[:, 1]
plt.hist(y_pred_prob, bins=8)
plt.xlim(0, 1)
plt.title('Pima Indian Diabetes Study Histogram')
plt.xlabel('Predicted Probabilities Diffusion')
plt.ylabel('Frequency')

# Adjusting the threshold to 0.1
y_pred_class = binarize([y_pred_prob], 0.1)[0]
# Data Visualisation of ROC curve :
# equilibrium between True (tpr) and False (fpr) Positive Rate

import numpy as np
line = np.linspace(0, 1, 2)
plt.plot(line, 'r')

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for Diabetes Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

# level of predictive precision
roc_auc_score(y_test, y_pred_prob)

### logistic regression
from sklearn.linear_model import LogisticRegression
logit1=LogisticRegression()
logit1.fit(X,y)

logit1.score(X,y)

#True positive
trueInput=Diabetes.ix[Diabetes['Outcome']==1].iloc[:,:8]
trueOutput=Diabetes.ix[Diabetes['Outcome']==1].iloc[:,8]
##True positive rate
np.mean(logit1.predict(trueInput)==trueOutput)
##Return around 55%

##True negative
falseInput=Diabetes.ix[Diabetes['Outcome']==0].iloc[:,:8]
falseOutput=Diabetes.ix[Diabetes['Outcome']==0].iloc[:,8]
##True negative rate
np.mean(logit1.predict(falseInput)==falseOutput)
##Return around 90%

###Confusion matrix with sklearn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
confusion_matrix(logit1.predict(inputData),outputData)

##Computing false and true positive rates
fpr, tpr,_=roc_curve(logit1.predict(inputData),outputData,drop_intermediate=False)

import matplotlib.pyplot as plt
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

#### decision tree
#step4 use training dataset to apply algorithm 
import seaborn as sns
model = DecisionTreeClassifier(max_depth=4, random_state=0)
tree_= model.fit(feature,target)
test_input=test[test.columns[0:8]]
expected = test["Outcome"]
#print("*******************Input******************")
#print(test_input.head(2))
#print("*******************Expected******************")
#print(expected.head(2))
predicted = model.predict(test_input)
print(metrics.classification_report(expected, predicted))
conf = metrics.confusion_matrix(expected, predicted)
print(conf)
print("Decision Tree accuracy: ",model.score(test_input,expected))
dtreescore = model.score(test_input,expected)


label = ["0","1"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)
print (a)

#Feature Importance DecisionTreeClassifier
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
print("DecisionTree Feature ranking:")
for f in range(feature.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feat_names[indices[f]], importance[indices[f]]))
plt.figure(figsize=(15,5))
plt.title("DecisionTree Feature importances")
plt.bar(range(feature.shape[1]), importance[indices], color="y", align="center")
plt.xticks(range(feature.shape[1]), feat_names[indices])
plt.xlim([-1, feature.shape[1]])
plt.show()

## KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=21)
neigh.fit(feature,target)
knnpredicted = neigh.predict(test_input)
print(metrics.classification_report(expected, knnpredicted))
print(metrics.confusion_matrix(expected, knnpredicted))
print("KNN accuracy: ",neigh.score(test_input,expected))
knnscore=neigh.score(test_input,expected)

names_ = []
results_ = []
results_.append(dtreescore)
results_.append(knnscore)
names_.append("DT")
names_.append("KNN")

#ax.set_xticklabels(names)
res = pd.DataFrame()
res['y']=results_
res['x']=names_
ax = sns.boxplot(x='x',y='y',data=res)

import graphviz 
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
dot_data=StringIO()
dot_data = export_graphviz(model, out_file = None, feature_names=feat_names, class_names=target_classes, 
                filled=True, rounded=True, special_characters=True) 

graph = pydotplus.graph_from_dot_data(dot_data)
print(dot_data)
Image(graph.create_png())
#graph.write_pdf("diabetes.pdf")

#Evaluation DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import random

fpr,tpr,thres = roc_curve(expected, predicted)
roc_auc = auc(fpr, tpr)
plt.title('DecisionTreeClassifier-Receiver Operating Characteristic Test Data')
plt.plot(fpr, tpr, color='green', lw=2, label='DecisionTree ROC curve (area = %0.2f)' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#KNeighborsClassifier-ROC curve
kfpr,ktpr,kthres = roc_curve(expected, knnpredicted)
kroc_auc = auc(kfpr, ktpr)
plt.title('KNeighborsClassifier- Receiver Operating Characteristic')
plt.plot(kfpr, ktpr, color='darkorange', lw=2, label='KNeighbors ROC curve (area = %0.2f)' % kroc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


