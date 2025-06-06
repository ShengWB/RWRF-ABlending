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
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from rrf import RRandomForest
from rwrf import RWRandomForest
from wrf import WRandomForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings(action='ignore')  # 忽略告警
# 读取数据
df = pd.read_excel('wheat_C_riskpre.xlsx')
# 构建特征和标签
X = df.drop(columns=['cvar'])  # 构建特征
y = df['cvar']  # 构建标签
# 数据集拆分：划分80%训练集和20%测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
py_test=pd.DataFrame(y_test)
py_test.to_excel('wheat_y_test.xlsx',index=False)
#SVM
SVR_model=SVR(kernel='linear',C=1.0, epsilon=0.1,gamma='scale')
SVR_model.fit(X_train,y_train)
sy_pre = SVR_model.predict(X_test)
pr=pd.DataFrame(sy_pre)
pr.to_excel('wheat_y_pred_svr.xlsx', index=False)
#BPNN
BP_model=MLPRegressor(hidden_layer_sizes=(10,4),activation='relu',solver='adam',alpha=0.01,max_iter=500)
BP_model.fit(X_train,y_train)
by_pre = BP_model.predict(X_test)
pr=pd.DataFrame(by_pre)
pr.to_excel('wheat_y_pred_bpnn.xlsx', index=False)
#GBT
gbt_model=GradientBoostingRegressor(learning_rate=0.01,max_depth=3,n_estimators=100,random_state=None)
gbt_model.fit(X_train,y_train)
gy_pre = gbt_model.predict(X_test)
pr=pd.DataFrame(gy_pre)
pr.to_excel('wheat_y_pred_gbt.xlsx', index=False)
#DT
DT_model=DecisionTreeRegressor(splitter='random',random_state=None)
DT_model.fit(X_train,y_train)
dy_pre = DT_model.predict(X_test)
pr=pd.DataFrame(dy_pre)
pr.to_excel('wheat_y_pred_dt.xlsx', index=False)
#KNN
KNN_model=KNeighborsRegressor(n_neighbors=5, weights='uniform',algorithm='auto',leaf_size=30, p=2, metric='minkowski', metric_params=None)
KNN_model.fit(X_train,y_train)
ky_pre = KNN_model.predict(X_test)
pr=pd.DataFrame(ky_pre)
pr.to_excel('wheat_y_pred_knn.xlsx', index=False)
#Cab
cat_model=CatBoostRegressor(iterations=100,learning_rate=0.01,depth=6,random_seed=None)
cat_model.fit(X_train,y_train)
cy_pre = cat_model.predict(X_test)
pr=pd.DataFrame(cy_pre)
pr.to_excel('wheat_y_pred_cab.xlsx', index=False)
#Ada
Ada_model=AdaBoostRegressor(learning_rate=0.01,n_estimators=100,random_state=None)
Ada_model.fit(X_train,y_train)
ay_pre = Ada_model.predict(X_test)
pr=pd.DataFrame(ay_pre)
pr.to_excel('wheat_y_pred_ada.xlsx', index=False)
#Xgb
Xgb_model=XGBRegressor(max_depth=4,learning_rate=0.03,n_estimators=100,random_state=None)
Xgb_model.fit(X_train,y_train)
xy_pre = Xgb_model.predict(X_test)
pr=pd.DataFrame(xy_pre)
pr.to_excel('wheat_y_pred_xgb.xlsx', index=False)

'''5折stacking'''
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=0)  # 5折交叉验证
# 定义基本模型
base_models = [
    MLPRegressor(hidden_layer_sizes=(10,4),activation='relu',solver='adam',alpha=0.01,max_iter=500),
    SVR(kernel='linear',C=1.0, epsilon=0.1,gamma='scale'),
    GradientBoostingRegressor(learning_rate=0.01,max_depth=3,n_estimators=100,random_state=None),
    DecisionTreeRegressor(splitter='random',random_state=None),
    KNeighborsRegressor(n_neighbors=5, weights='uniform',algorithm='auto',leaf_size=30, p=2, metric='minkowski', metric_params=None),
    CatBoostRegressor(iterations=100,learning_rate=0.01,depth=6,random_seed=None),
    AdaBoostRegressor(learning_rate=0.01,n_estimators=100,random_state=None),
    XGBRegressor(max_depth=4,learning_rate=0.03,n_estimators=100,random_state=None)
]
'''切分训练数据集为d1,d2两部分'''
X_d1, X_d2, y_d1, y_d2 = train_test_split(X_train, y_train, test_size=0.5, random_state=None)

def huber_loss(y_true, y_pred, delta=1.0):
    residual = np.abs(y_true - y_pred)
    condition = residual < delta
    loss = np.where(condition, 0.5 * np.square(residual), delta * (residual - 0.5 * delta))
    return loss
def PSI_indicators(y_test,y_pred):
    zong1=y_test-y_pred
    zong2=np.log(y_test/y_pred)
    core=sum(zong1*zong2)
    return core
def acc_score(y_test,y_pred,delta):
    huber_error=huber_loss(y_test,y_pred,delta)
    huber=sum(huber_error)/len(y_test)
    zhuber=1-huber
    return zhuber

# 训练基本模型并记录误差
acca_score=[]
psi_score=[]
zonghe_score=[]
for base_model in base_models:
    base_model.fit(X_train, y_train)
    y_pred= base_model.predict(X_test)
    accurate_score=acc_score(y_pred,y_test,delta=1.0)
    acca_score.append(accurate_score)
    psi_score1=PSI_indicators(y_pred,y_test)
    psi_score.append(psi_score1)
    gama = 0.5
    zonghe_score1=gama*psi_score1+(1-gama)*accurate_score
    zonghe_score.append(zonghe_score1)
print("Acc",acca_score)
print("PSI", psi_score)
print("MCP",zonghe_score)

# 对误差进行排序并选择排名靠前的模型
sorted_models=[model for _, model in sorted(zip(zonghe_score, base_models))]
selected_models = sorted_models[-3:]
print(sorted_models)
print(selected_models)
model_names={
    "Base_model1":selected_models[0],
    "Base_model2":selected_models[1],
    "Base_model3":selected_models[2]

}
Base_model1=model_names["Base_model1"]
Base_model2=model_names["Base_model2"]
Base_model3=model_names["Base_model3"]


t1=Base_model1.fit(X_d1,y_d1)
t2=Base_model2.fit(X_d1,y_d1)
t3=Base_model3.fit(X_d1,y_d1)
#t4=Base_model4.fit(X_d1,y_d1)
#t5=Base_model5.fit(X_d1,y_d1)
#t6=Base_model6.fit(X_d1,y_d1)
#t7=Base_model7.fit(X_d1,y_d1)
#t8=Base_model8.fit(X_d1,y_d1)
y1=Base_model1.predict(X_d2)
y2=Base_model2.predict(X_d2)
y3=Base_model3.predict(X_d2)
#y4=Base_model4.predict(X_d2)
#y5=Base_model5.predict(X_d2)
#y6=Base_model6.predict(X_d2)
#y7=Base_model7.predict(X_d2)
#y8=Base_model8.predict(X_d2)
test_predict1=Base_model1.predict(X_test)
test_predict2=Base_model2.predict(X_test)
test_predict3=Base_model3.predict(X_test)
test_predict1 = pd.DataFrame(test_predict1)
test_predict2 = pd.DataFrame(test_predict2)
test_predict3 = pd.DataFrame(test_predict3)
test_predict1.to_excel('y1.xlsx', index=False)
test_predict2.to_excel('y2.xlsx', index=False)
test_predict3.to_excel('y3.xlsx', index=False)
#test_predict4=Base_model4.predict(X_test)
#test_predict5=Base_model5.predict(X_test)
#test_predict6=Base_model6.predict(X_test)
#test_predict7=Base_model7.predict(X_test)
#test_predict8=Base_model8.predict(X_test)

RF_train_oof=np.column_stack((y1, y2, y3))
RF_test_oof=np.column_stack((test_predict1, test_predict2, test_predict3))
# print(RF_test_oof.shape)
y_test=y_test.to_numpy()
y_d2=y_d2.to_numpy()

# 定义RF模型
if __name__ == '__main__':
    print("begin generate forest")
    rf=RandomForestRegressor(n_estimators=200, max_depth=3,criterion="mse")
    rf.fit(RF_train_oof,y_d2)
    rf_pred=rf.predict(RF_test_oof)
    pr1=pd.DataFrame(rf_pred)
    pr1.to_excel('wheat_y_pred_rf_blending.xlsx',index=False)
    rrf= RRandomForest(n_trees=120,max_depth=3,classifier=False,criterion="mse",n_feats=10)
    rrf.fit(RF_train_oof,y_d2)
    rrf_pred=rrf.predict(RF_test_oof)
    pr2=pd.DataFrame(rrf_pred)
    pr2.to_excel('wheat_y_pred_rrf_blending.xlsx',index=False)
    wrf= WRandomForest(n_trees=120,max_depth=3,classifier=False,criterion="mse",n_feats=10)
    wrf.fit(RF_train_oof,y_d2)
    wrf_pred=wrf.predict(RF_test_oof,y_test)
    pr3=pd.DataFrame(rrf_pred)
    pr3.to_excel('wheat_y_pred_wrf_blending.xlsx',index=False)
    rwrf = RWRandomForest(n_trees=200,max_depth=3,classifier=False,criterion="mse",n_feats=10)
    rwrf.fit(RF_train_oof,y_d2)
    # 可视化特征重要性
    importances = rf.feature_importances_
    rwrf_pred=rwrf.predict(RF_test_oof,y_test)
    pr4=pd.DataFrame(rwrf_pred)
    dr=pd.DataFrame(y_test)
    pr4.to_excel('wheat_y_pred_rwrf_blending.xlsx', index=False)
    # 多模型评估
    bp_mse = mean_squared_error(y_test, by_pre)
    bp_mae = mean_absolute_error(y_test, by_pre)
    bp_rmse = np.sqrt(bp_mse)
    bp_r2 = r2_score(y_test, by_pre)
    svm_mse = mean_squared_error(y_test, sy_pre)
    svm_mae = mean_absolute_error(y_test, sy_pre)
    svm_rmse = np.sqrt(svm_mse)
    svm_r2 = r2_score(y_test, sy_pre)
    gbt_mse = mean_squared_error(y_test, gy_pre)
    gbt_mae = mean_absolute_error(y_test, gy_pre)
    gbt_rmse = np.sqrt(gbt_mse)
    gbt_r2 = r2_score(y_test, gy_pre)
    dt_mse = mean_squared_error(y_test, dy_pre)
    dt_mae = mean_absolute_error(y_test, dy_pre)
    dt_rmse = np.sqrt(dt_mse)
    dt_r2 = r2_score(y_test, dy_pre)
    knn_mse = mean_squared_error(y_test, ky_pre)
    knn_mae = mean_absolute_error(y_test, ky_pre)
    knn_rmse = np.sqrt(knn_mse)
    knn_r2 = r2_score(y_test, ky_pre)
    cab_mse = mean_squared_error(y_test, cy_pre)
    cab_mae = mean_absolute_error(y_test, cy_pre)
    cab_rmse = np.sqrt(cab_mse)
    cab_r2 = r2_score(y_test, cy_pre)
    ada_mse = mean_squared_error(y_test, ay_pre)
    ada_mae = mean_absolute_error(y_test, ay_pre)
    ada_rmse = np.sqrt(ada_mse)
    ada_r2 = r2_score(y_test, ay_pre)
    xgb_mse = mean_squared_error(y_test, xy_pre)
    xgb_mae = mean_absolute_error(y_test, xy_pre)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_r2 = r2_score(y_test, xy_pre)
    rf_ble_mse = mean_squared_error(y_test, rf_pred)
    rf_ble_mae = mean_absolute_error(y_test, rf_pred)
    rf_ble_rmse = np.sqrt(rf_ble_mse)
    rf_ble_r2 = r2_score(y_test, rf_pred)

    rrf_ble_mse = mean_squared_error(y_test, rrf_pred)
    rrf_ble_mae = mean_absolute_error(y_test, rrf_pred)
    rrf_ble_rmse = np.sqrt(rrf_ble_mse)
    rrf_ble_r2 = r2_score(y_test, rrf_pred)

    wrf_ble_mse = mean_squared_error(y_test, wrf_pred)
    wrf_ble_mae = mean_absolute_error(y_test, wrf_pred)
    wrf_ble_rmse = np.sqrt(wrf_ble_mse)
    wrf_ble_r2 = r2_score(y_test, wrf_pred)

    rwrf_ble_mse=mean_squared_error(y_test,rwrf_pred)
    rwrf_ble_mae=mean_absolute_error(y_test,rwrf_pred)
    rwrf_ble_rmse=np.sqrt(rwrf_ble_mse)
    rwrf_ble_r2=r2_score(y_test,rwrf_pred)
    Table = [["BPNN", bp_mse, bp_mae, bp_rmse, bp_r2], ["SVM", svm_mse, svm_mae, svm_rmse, svm_r2],
             ["GBDT", gbt_mse, gbt_mae, gbt_rmse, gbt_r2], ["DT", dt_mse, dt_mae, dt_rmse, dt_r2],
             ["KNN", knn_mse, knn_mae, knn_rmse, knn_r2], ["Catboost", cab_mse, cab_mae, cab_rmse, cab_r2],
             ["Adaboost", ada_mse, ada_mae, ada_rmse, ada_r2], ["Xgboost", xgb_mse, xgb_mae, xgb_rmse, xgb_r2],
             ["RF-Blending", rf_ble_mse, rf_ble_mae, rf_ble_rmse, rf_ble_r2],["RRF-Blending", rf_ble_mse, rf_ble_mae, rf_ble_rmse, rf_ble_r2],
             ["WRF-Blending", rf_ble_mse, rf_ble_mae, rf_ble_rmse, rf_ble_r2],["RWRF-Blending",rwrf_ble_mse, rwrf_ble_mae, rwrf_ble_rmse, rwrf_ble_r2]]
    df = pd.DataFrame(Table, columns=["model_name", "MSE", "MAE", "RMSE", "R2"])


print(df)
