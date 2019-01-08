import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def loss_function(x, y):
    """损失函数
    返回x-y的L2范数
    """
    # out = 0
    # for i in range(len(y)):
    #     out += (x[i] - y[i]) ** 2 / 2 / len(y)
    # return out
    temp = x - y
    return np.linalg.norm(temp) / 2 / len(y)


class LinearRegression:
    def __init__(self, n):
        """初始化
        初始化n+1个参数，用n+1维向量表示，n维元数
        """
        self.parameters = np.random.randn(n + 1).reshape(-1, 1)
        self._paras_size = n + 1

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        """预测函数
        x为m*n维矩阵
        x'为x的增广矩阵，尺寸为m*(n+1)
        返回输入x'各分量的线性累加
        """
        # out = self.parameters[-1]
        # for i, ix in enumerate(x):
        #     out = ix * self.parameters[i]
        # return out
        x = np.c_[x, np.ones(x.shape[0])]  # 将输入变为增广向量，即将[x1,x2,...,xn]变为[x1,x2,...,xn,1]
        return x @ self.parameters

    def fit(self, x, y):
        """训练参数
        x为训练集自变量，为m*n维矩阵
        y为训练集目标，为m维向量
        n为元数，m为样本量
        """
        learning_rate = 0.01
        pre_y = self.predict(x)
        for i in range(self._paras_size - 1):
            self.parameters[i] -= learning_rate * np.mean((pre_y - y) * x[:, i:i + 1])
        self.parameters[-1] -= learning_rate * np.mean(pre_y - y)

        # learning_rate = 0.01
        # m = len(y)  # 样本量
        # pre_y = self.predict(x)
        # cost = learning_rate * (pre_y - y) / len(y)  # 此处为损失函数对pre_y求导后的结果
        # x = np.np.c_[x, np.ones(m)]  # 将输入变为增广向量
        # self.parameters = x @ cost


if __name__ == "__main__":
    n = 1
    LR_test = LinearRegression(n)
    data = np.arange(-99, 100).reshape(-1, n) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        data, 10 + data * 5 + 0.1 * np.random.randn(199).reshape(-1, n),
        test_size=0.25, random_state=0)
    plt.scatter(X_train, y_train)

    for i in range(1000):
        LR_test.fit(X_train, y_train)
        if not i % 100:
            print(loss_function(LR_test(X_train), y_train))
            plt.plot(X_test, LR_test(X_test))
            plt.scatter(X_train, y_train)
            plt.show()
