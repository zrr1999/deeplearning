from Linear_Regression import LinearRegression, mean_square_error
import numpy as np


class PolynomialRegression(LinearRegression):
    def __init__(self, n, *f):
        self.num_f = len(f) + 1  # 除了输入的函数集还有线性函数
        super(PolynomialRegression, self).__init__(n * self.num_f)
        self.f = f

    def __call__(self, x):
        t = x[:]  # 保存初始输入值
        for temp_f in self.f:
            x = np.c_[x, temp_f(t)]
        return self._predict(x)

    def fit(self, x, y):
        t = x[:]  # 保存初始输入值
        for temp_f in self.f:
            x = np.c_[x, temp_f(t)]
        super(PolynomialRegression, self).fit(x, y)


if __name__ == "__main__":
    PR_test = PolynomialRegression(1, np.exp, np.log)
    X = np.array([[1],[2],[3]])
    Y = np.exp(X)
    print(PR_test(X))
    for i in range(1000):
        PR_test.fit(X, Y)
        print(mean_square_error(PR_test(X), Y))
    print(PR_test(X))
