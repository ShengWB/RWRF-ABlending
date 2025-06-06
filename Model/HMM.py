import numpy as np
from scipy.linalg import norm, pinv
import xlrd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import csv
from hmmlearn import hmm
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import pandas as pd

# 读取数据
df = pd.read_excel('wheat_C_riskpre.xlsx')
# 构建特征和标签
X = df.drop(columns=['cvar'])  # 构建特征
y = df['cvar']  # 构建标签
# 数据集拆分：划分80%训练集和20%测试集
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=None)
#py_test=pd.DataFrame(y_test)
#py_test.to_excel('wheat_y_test.xlsx',index=False)
# 初始化模型
model = hmm.GaussianHMM(n_components=4, n_iter=1000,tol=1e-3,verbose=True)
# 拟合模型，传入观测序列和对应的长度
lenth=len(X)
model.fit(X,lengths=[lenth])
logprob, states = model.decode([X], algorithm="viterbi")
print("Predicted states:", states[0])
print("Log likelihood of the sequence:", logprob)
# 预测下一个观测
predicted_obs = model.predict_proba(X)[0][-1]
print("Predicted next observation:", np.argmax(predicted_obs))
HMM_mse=mean_squared_error(predicted_obs,y)
HMM_mae=mean_absolute_error(predicted_obs,y)
HMM_rmse=np.sqrt(HMM_mse)
HMM_r2=r2_score(predicted_obs,y)
print(HMM_mse,HMM_mae,HMM_rmse,HMM_r2)
