import numpy as np


def mean_square_error(x, y):
    """均方误差函数
    返回x-y的L2范数
    """
    # out = 0
    # for i in range(len(y)):
    #     out += (x[i] - y[i]) ** 2 / 2 / len(y)
    # return out

    temp = x - y
    return np.linalg.norm(temp) / 2 / len(y)


class LinearRegression:
    def __init__(self, n, learning_rate=0.1):
        """初始化
        初始化n+1个参数，用n+1维列向量表示，n为元数
        """
        self.parameters = np.random.randn(n + 1).reshape(-1, 1)
        self._paras_size = n + 1
        self.learning_rate = learning_rate

    def __call__(self, x):
        return self._predict(x)

    def _predict(self, x):
        """预测函数
        x为m*n矩阵
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
        x为训练集自变量，为m*n矩阵
        y为训练集目标，为m*1矩阵
        n为元数，m为样本量
        """
        # learning_rate = 0.01
        # pre_y = self.predict(x)
        # for _i in range(self._paras_size - 1):
        #     self.parameters[_i] -= learning_rate * np.mean((pre_y - y) * x[:, _i:_i + 1])
        # self.parameters[-1] -= learning_rate * np.mean(pre_y - y)

        m = len(y)  # 样本量
        pre_y = self._predict(x)
        cost = self.learning_rate * (pre_y - y) / m  # 此处为均方误差函数对pre_y求导后的结果
        x = np.c_[x, np.ones(m)]  # 将输入变为增广向量
        self.parameters -= x.T @ cost


if __name__ == "__main__":
    n = 2
    LR_test = LinearRegression(n)
    X = np.array([[1, 2], [2, 3]])
    Y = np.array([[3], [5]])
    for i in range(5000):
        LR_test.fit(X, Y)
        if not (i + 1) % 50:
            print("进度：{:.0f}%，当前损失：{}".format(i / 50, mean_square_error(LR_test(X), Y)))
    print(LR_test(X))
    # n = 2
    # LR_test = LinearRegression(n)
    # datasets = np.arange(-99, 100).reshape(-1, n) / 100
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     datasets, 10 + datasets * 5 + 0.1 * np.random.randn(199).reshape(-1, n),
    #     test_size=0.25, random_state=0)
    # plt.scatter(X_train, y_train)
    #
    # for i in range(1000):
    #     LR_test.fit(X_train, y_train)
    #     if not i % 100:
    #         print(loss_function(LR_test(X_train), y_train))
    #         plt.scatter(X_train, y_train)
    #         plt.plot(X_test, LR_test(X_test))
    #         plt.show()
