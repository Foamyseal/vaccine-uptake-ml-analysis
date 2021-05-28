import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataset = np.loadtxt('upd-combined-education-vaccine-set.csv', delimiter=',')
print(dataset[1])
print(dataset[2])
print(dataset[6])
X = dataset[:,2:9]
Y = dataset[:,1]
print(X)
print(Y)

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.15)
model = xgb.XGBRegressor()

model.fit(xtrain, ytrain)
print(model.feature_importances_)
xgb.plot_importance(model)
plt.show()
score = model.score(xtrain, ytrain)
print("Training score: ", score)

scores = cross_val_score(model, xtrain, ytrain,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(model, xtrain, ytrain, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

ypred = model.predict(xtest)
mse = mean_squared_error(ytest, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))

x_ax = range(len(ytest))
plt.plot(x_ax, ytest, label="original")
plt.plot(x_ax, ypred, label="predicted")
plt.title("test")
plt.legend()
plt.show()