from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from pathlib import Path
# 读取数据
df = pd.read_excel('wheat_C_riskpre.xlsx')
# 构建特征和标签
target = 'cvar'
if "Set" not in df.columns:
    df["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(df.shape[0],))
unused_feat = ['Set']
features = [ col for col in df.columns if col not in unused_feat+[target]]
#cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
#cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# define your embedding sizes : here just a random choice
cat_emb_dim = [5, 4, 3, 6, 2, 2, 1, 10]
train_indices = df[df.Set=="train"].index
valid_indices = df[df.Set=="valid"].index
test_indices = df[df.Set=="test"].index

clf=TabNetRegressor()
X_train = df[features].values[train_indices]
y_train = df[target].values[train_indices].reshape(-1, 1)

X_valid = df[features].values[valid_indices]
y_valid = df[target].values[valid_indices].reshape(-1, 1)

X_test = df[features].values[test_indices]
y_test = df[target].values[test_indices].reshape(-1, 1)
max_epochs = 100 if not os.getenv("CI", False) else 2
from pytorch_tabnet.augmentations import RegressionSMOTE
aug = RegressionSMOTE(p=0.2)
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
    max_epochs=max_epochs,
    patience=50,
    batch_size=16, virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    augmentations=aug, #aug
)
preds = clf.predict(X_test)
y_true = y_test
Tab_mse = mean_squared_error(y_pred=preds, y_true=y_true)
Tab_mae = mean_absolute_error(y_pred=preds, y_true=y_true)
Tab_rmse = np.sqrt(Tab_mse)
Tab_r2 = r2_score(y_pred=preds, y_true=y_true)
print(Tab_mse,Tab_mae,Tab_rmse,Tab_r2)
