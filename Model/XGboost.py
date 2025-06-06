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
Xgb_model=XGBRegressor(max_depth=4,learning_rate=0.03,n_estimators=100,random_state=None,reg_alpha=0.1,reg_lambda=0.5)
Xgb_model.fit(X_train,y_train)
xy_pre = Xgb_model.predict(X_test)
pr=pd.DataFrame(xy_pre)
pr.to_excel('wheat_y_pred_xgb.xlsx', index=False)
xgb_mse=mean_squared_error(y_test,xy_pre)
xgb_mae=mean_absolute_error(y_test,xy_pre)
xgb_rmse=np.sqrt(xgb_mse)
xgb_r2=r2_score(y_test,xy_pre)
Table= [["Xgboost",xgb_mse,xgb_mae,xgb_rmse,xgb_r2]]
df = pd.DataFrame(Table, columns=["model_name", "MSE", "MAE","RMSE","R2"])
print(df)