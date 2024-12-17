import numpy as np

OPV_dataset = np.loadtxt('OPV.txt')
data = OPV_dataset[:,[0, 1, 2]]
target = OPV_dataset[:,6]
n=len(open('OPV.txt').readlines())

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
loo = LeaveOneOut()
print("MachineLearning starting:")

print("linear regression:")
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
prediction = cross_val_predict(lr, data, target, cv=loo)
mae = mean_absolute_error(target, prediction)
print("MAE: {:.2f}".format(mae))
rmse = np.sqrt(mean_squared_error(target, prediction))
print("RMSE: {:.2f}".format(rmse))
p_value = pearsonr(target, prediction)
print("Pearson's correlation coefficient: ", p_value)
with open('Prediction-lr.txt', 'wt') as f:
    for i in range(n):
        print("{:6.2f} {:6.2f}".format(target[i],prediction[i]),file=f)

print("k-NN regression:")
from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors=10)
prediction = cross_val_predict(knnr, data, target, cv=loo)
mae = mean_absolute_error(target, prediction)
print("MAE: {:.2f}".format(mae))
rmse = np.sqrt(mean_squared_error(target, prediction))
print("RMSE: {:.2f}".format(rmse))
p_value = pearsonr(target, prediction)
print("Pearson's correlation coefficient: ", p_value)
with open('Prediction-knnr.txt', 'wt') as f:
    for i in range(n):
        print("{:6.2f} {:6.2f}".format(target[i],prediction[i]),file=f)

print("gradient boosting regression:")
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=0, max_depth=3,learning_rate=0.04)
prediction = cross_val_predict(gbr, data, target, cv=loo)
mae = mean_absolute_error(target, prediction)
print("MAE: {:.2f}".format(mae))
rmse = np.sqrt(mean_squared_error(target, prediction))
print("RMSE: {:.2f}".format(rmse))
p_value = pearsonr(target, prediction)
print("Pearson's correlation coefficient: ", p_value)
gbr.fit(data, target)
print("descriptor importance:", gbr.feature_importances_)
with open('Prediction-gbrt.txt', 'wt') as f:
    for i in range(n):
        print("{:6.2f} {:6.2f}".format(target[i],prediction[i]),file=f)
