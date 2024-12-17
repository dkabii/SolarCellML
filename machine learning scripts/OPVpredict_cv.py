import numpy as np

OPV_dataset = np.loadtxt('OPV.txt')
data = OPV_dataset[:,[0, 1, 2]]
target = OPV_dataset[:,6]
n=len(open('OPV.txt').readlines())

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
from sklearn.model_selection import cross_val_predict
from scipy.stats import pearsonr
print("MachineLearning starting:")

print("linear regression:")
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
scores = cross_val_score(lr, data, target, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
prediction = cross_val_predict(lr, data, target, cv=kfold)
p_value = pearsonr(target, prediction)
print("Pearson's correlation coefficient: ", p_value)
with open('Prediction-lr.txt', 'wt') as f:
    for i in range(n):
        print("{:6.2f} {:6.2f}".format(target[i],prediction[i]),file=f)
        
print("k-NN regression:")
from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors=10)
scores = cross_val_score(knnr, data, target, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
prediction = cross_val_predict(knnr, data, target, cv=kfold)
p_value = pearsonr(target, prediction)
print("Pearson's correlation coefficient: ", p_value)
with open('Prediction-knnr.txt', 'wt') as f:
    for i in range(n):
        print("{:6.2f} {:6.2f}".format(target[i],prediction[i]),file=f)
        
print("gradient boosting regression:")
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=0, max_depth=3,learning_rate=0.04)
scores = cross_val_score(gbr, data, target, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
prediction = cross_val_predict(gbr, data, target, cv=kfold)
p_value = pearsonr(target, prediction)
print("Pearson's correlation coefficient: ", p_value)
with open('Prediction-gbrt.txt', 'wt') as f:
    for i in range(n):
        print("{:6.2f} {:6.2f}".format(target[i],prediction[i]),file=f)
