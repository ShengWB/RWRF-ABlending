import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
# 读取数据
X = pd.read_excel('wheat_C_X.xlsx',header=None)
y = pd.read_excel('wheat_C_Y.xlsx',header=None)
# 数据集拆分：划分80%训练集和20%测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

class RBFNet:
    def __init__(self, n_hidden, sigma=1.0):
        self.n_hidden = n_hidden
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _gaussian(self, x, center):
        return np.exp(-self.sigma * np.linalg.norm(x - center) ** 2)

    def _calculate_centers(self, X):
        self.centers = X.iloc[np.random.choice(X.shape[0], self.n_hidden, replace=False)]

    def _calculate_weights(self, X, y):
        phi = np.zeros((X.shape[0], self.n_hidden))

        for i, c in enumerate(self.centers):
            phi[:, i] = np.array([self._gaussian(x, c) for x in X])

        self.weights = np.dot(np.linalg.pinv(phi), y)

    def fit(self, X, y):
        self._calculate_centers(X)
        self._calculate_weights(X, y)

    def predict(self, X):
        y_pred = np.zeros((X.shape[0],))

        for i, x in enumerate(X):
            phi = np.array([self._gaussian(x, c) for c in self.centers])
            y_pred[i] = np.dot(phi, self.weights)

        return y_pred

# 训练RBF神经网络
rbf = RBFNet(n_hidden=10, sigma=1.0)
rbf.fit(X_train, y_train)

# 预测测试集
y_pred = rbf.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)