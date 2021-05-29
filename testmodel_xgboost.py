import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


dataset = np.loadtxt('combined-education-vaccine-set-(5).csv', delimiter=',',  encoding='utf-8-sig')
X = dataset[:,2:65]
Y = dataset[:,1]
print(X[0])
print(Y[0])

seed = 7 
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = mean_squared_error(y_test, y_pred)
print("MSE:", accuracy)