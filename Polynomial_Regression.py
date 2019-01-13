from Linear_Regression import LinearRegression, mean_square_error
import numpy as np


class PolynomialRegression(LinearRegression):
    def __init__(self, n, *f, **kwargs):
        self.num_f = len(f) + 1  # 除了输入的函数集还有线性函数
        super(PolynomialRegression, self).__init__(n * self.num_f, **kwargs)
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
    PR_test = PolynomialRegression(1, np.exp, np.log, learning_rate=0.01)
    X = np.array([[1], [2], [3]])
    Y = 5*np.exp(X)+X*np.log(X)+100
    for i in range(5000):
        PR_test.fit(X, Y)
        if not (i+1)%50:
            print("进度：{:.0f}%，当前损失：{}".format(i/50,mean_square_error(PR_test(X), Y)))
    print(PR_test(X))
