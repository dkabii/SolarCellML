import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

df = pd.read_csv('OPV.txt', delimiter='\s+', header=None)
df.columns = [['Eg', 'EDA', 'EST', 'VOC', 'JSC', 'FF', 'PCE']]
print('correlation')
corr = df.corr()
print(corr)
#sns.heatmap(corr,annot=True)
#plt.show()


features = df.loc[:,['Eg', 'EDA', 'EST', 'JSC', 'FF']]
features2 = df.iloc[:, :-1]
target = df['PCE'].values.ravel()

kfold = KFold(n_splits=5, shuffle=True, random_state=0)

print('radial basis function svm with all features except VOC')
svr = SVR(kernel='rbf')
scores = cross_val_score(svr, features, target, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))

print('radial basis function svm with all features')
svr = SVR(kernel='rbf')
scores = cross_val_score(svr, features2, target, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))


print('polynomial svm with all features except VOC')
svr = SVR(kernel='poly')
scores = cross_val_score(svr, features, target, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))

print('polynomial svm with all features')
svr = SVR(kernel='poly')
scores = cross_val_score(svr, features2, target, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))

print('linear svm with all features except VOC')
svr = SVR(kernel='linear')
scores = cross_val_score(svr, features, target, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))

print('linear svm with all features')
svr = SVR(kernel='linear')
scores = cross_val_score(svr, features2, target, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))









#mean accuracy of models from OPVpredict_cv.py
model_names = ['Linear regression','kNN','Gradient Boosting']
model_scores = [0.5,0.61,0.65]