# coding=utf8
from typing import List, Union

import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings(action='ignore')  # 忽略告警
# 读取数据
df = pd.read_excel('wheat_C_riskpre.xlsx')
# 构建特征和标签
X = df.drop(columns=['cvar'])  # 构建特征
y = df['cvar']  # 构建标签
# 数据集拆分：划分80%训练集和20%测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 进行数据拆分
'''5折stacking'''
ntrain=X_train.shape[0]
ntest=X_test.shape[0]
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=0)  # 5折交叉验
'''切分训练数据集为d1,d2两部分'''
X_d1, X_d2, y_d1, y_d2 = train_test_split(X_train, y_train, test_size=0.5, random_state=2017)

# 定义一个扩展类 SklearnHelper
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):  # 定义初始化方法
        params['random_state'] = seed
        self.clf = clf(**params)

    def predict(self, x):  # 定义预测方法
        return self.clf.predict(x)

    def train(self, x_train, y_train):  # 定义训练方法
        self.clf.fit(x_train, y_train)

    def fit(self, x, y):  # 定义拟合方法
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):  # 定义特征重要性方法
        return self.clf.fit(x, y).feature_importances_
Base_model1=XGBRegressor(max_depth=4,learning_rate=0.1,n_estimators=200,random_state=None,reg_lambda=1,reg_alpha=0.1)
Base_model2=LGBMRegressor(learning_rate=0.1,n_estimators=200,max_depth=4)
train = X_train
x_train = train.values
x_test = X_test.values
y_train = y_train.values
#定义训练方法
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))  # 训练数据初始化
    oof_test = np.zeros((ntest,))  # 训练数据初始化
    oof_test_skf = np.empty((NFOLDS, ntest))  # 定义空数组

    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):  # 循环
        x_tr = x_train[train_index]  # 训练集-特征
        y_tr = y_train[train_index]  # 训练集-标签
        x_te = x_train[test_index]  # 测试集-特征

        clf.fit(x_tr, y_tr)  # 训练

        oof_train[test_index] = clf.predict(x_te)  # 预测
        oof_test_skf[i, :] = clf.predict(x_test)  # 预测

    oof_test[:] = oof_test_skf.mean(axis=0)  # 取均值
    return oof_train, oof_test  # 返回训练集和测试集预测数据
Base_model1_train,Base_model1_test=get_oof(Base_model1,x_train,y_train,x_test)
Base_model2_train,Base_model2_test=get_oof(Base_model2,x_train,y_train,x_test)
gru_x_train=np.column_stack((Base_model1_train,Base_model2_train))
gru_x_test=np.column_stack((Base_model1_test,Base_model2_test))
GRU_x_train=np.column_stack((gru_x_train,X_train))
GRU_x_test=np.column_stack((gru_x_test,X_test))
# 定义LSTM模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(14, 1)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(32, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
print(model.summary())
model.fit(GRU_x_train,y_train,batch_size =16,epochs =200,verbose=1)
y_prediction = model.predict(GRU_x_test)
pr=pd.DataFrame(y_prediction)
dr=pd.DataFrame(y_test)
mse=mean_squared_error(y_test,y_prediction)
mae=mean_absolute_error(y_test,y_prediction)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_prediction)
print(mse,mae,rmse,r2)
