from Linear_Regression import LinearRegression, mean_square_error
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy(x, y):
    """交叉熵函数
    返回x与y的交叉熵
    """
    return -y.T @ np.log(x) - (1 - y).T @ np.log(1 - x)


class LogisticRegression(LinearRegression):
    def __init__(self, n):
        super(LogisticRegression, self).__init__(n)

    def __call__(self, x):
        x = self._predict(x)
        return sigmoid(x)

    def fit(self, x, y):
        learning_rate = 0.1
        m = len(y)  # 样本量
        t = self._predict(x)
        pre_y = sigmoid(t)
        cost = learning_rate * ((y - 1) / (pre_y - 1) - y / pre_y) / m  # 此处为交叉熵函数对pre_y求导后的结果
        cost *= pre_y * (1 - pre_y)  # 此处为pre_y对t求导后的结果
        x = np.c_[x, np.ones(m)]  # 将输入变为增广向量
        self.parameters -= x.T @ cost


if __name__ == "__main__":
    n = 2
    LR_test = LogisticRegression(n)
    X = np.array([[3, 3], [2, 2]])
    Y = np.array([[0], [1]])
    print(LR_test(X))
    for i in range(500):
        LR_test.fit(X, Y)
        print(cross_entropy(LR_test(X), Y))
    print(LR_test(X))
