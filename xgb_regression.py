import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.figure_factory as ff
from numba import jit, cuda
import shap


dataset = np.loadtxt('upd-combined-education-vaccine-set-final.csv', delimiter=',')
fips = [int(i) for i in dataset[:,0].tolist()]

X = dataset[:,2:85]
Y = dataset[:,1]
print(fips)

@cuda.jit
def function(): 
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.33)

    model = xgb.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

    model.fit(xtrain, ytrain)
    print(model.feature_importances_)
    xgb.plot_tree(model)
    fig, ax = plt.subplots(1,1,figsize=(10, 20))
    xgb.plot_importance(model, ax=ax)
    plt.show()
    score = model.score(xtrain, ytrain)
    print("Training score: ", score)

function()

# scores = cross_val_score(model, xtrain, ytrain,cv=10)
# print("Mean cross-validation score: %.2f" % scores.mean())

# kfold = KFold(n_splits=10, shuffle=True)
# kf_cv_scores = cross_val_score(model, xtrain, ytrain, cv=kfold )
# print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

# ypred = model.predict(xtest)
# mse = mean_squared_error(ytest, ypred)
# print("MSE: %.2f" % mse)
# print("RMSE: %.2f" % (mse**(1/2.0)))

# x_ax = range(len(ytest))
# plt.plot(x_ax, ytest, label="original")
# plt.plot(x_ax, ypred, label="predicted")
# plt.title("test")
# plt.legend()
# plt.show()

# ypred = model.predict(X)
# mse = mean_squared_error(Y, ypred)
# print("MSE: %.2f" % mse)
# print("RMSE: %.2f" % (mse**(1/2.0)))
# kf_cv_scores = cross_val_score(model, X, Y, cv=kfold)
# print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
# x_ax = range(len(Y))
# plt.bar(x_ax, (Y*100), label="original")
# plt.bar(x_ax, (ypred*100), label="predicted")
# plt.title("vaccination uptake")
# plt.legend()
# plt.show()


# colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
#               "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
#               "#08519c","#0b4083","#08306b"]
# endpts = list(np.linspace(1, 100, len(colorscale) - 1))

# fig1 = ff.create_choropleth(
#     fips=fips, values=ypred*100,
#     binning_endpoints=endpts,
#     colorscale=colorscale,
#     show_state_data=False,
#     show_hover=True, centroid_marker={'opacity': 0},
#     asp=2.9, title='USA by Predicted Vaccination Uptake %',
#     legend_title='% predicted vaccinated'
# )

# fig1.layout.template = None
# fig1.show()

# colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
#               "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
#               "#08519c","#0b4083","#08306b"]

# fig2 = ff.create_choropleth(
#     fips=fips, values=Y*100,
#     binning_endpoints=endpts,
#     colorscale=colorscale,
#     show_state_data=False,
#     show_hover=True, centroid_marker={'opacity': 0},
#     asp=2.9, title='USA by Actual Vaccination Uptake %',
#     legend_title='% actual vaccinated'
# )

# fig2.layout.template = None
# fig2.show()

