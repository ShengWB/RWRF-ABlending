# coding=utf8
from typing import List, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings(action='ignore')  # 忽略告警
# 读取数据
Train = pd.read_excel('wheat_train.xlsx')
# 构建特征和标签
X_train = Train.drop(columns=['cvar'])  # 构建特征
y_train = Train['cvar']  # 构建标签
Test = pd.read_excel('wheat_test.xlsx')
X_test = Test.drop(columns=['cvar'])  # 构建特征
y_test = Test['cvar']  # 构建标签
#Xgb
KNN_model=KNeighborsRegressor(n_neighbors=6, weights='uniform',algorithm='auto',leaf_size=30, p=2, metric='minkowski', metric_params=None)
KNN_model.fit(X_train,y_train)
ky_pre = KNN_model.predict(X_test)
pr=pd.DataFrame(ky_pre)
pr.to_excel('wheat_y_pred_knn.xlsx', index=False)
knn_mse = mean_squared_error(y_test, ky_pre)
knn_mae = mean_absolute_error(y_test, ky_pre)
knn_rmse = np.sqrt(knn_mse)
knn_r2 = r2_score(y_test, ky_pre)
Table= [["KNN", knn_mse, knn_mae, knn_rmse, knn_r2]]
df = pd.DataFrame(Table, columns=["model_name", "MSE", "MAE","RMSE","R2"])
print(df)