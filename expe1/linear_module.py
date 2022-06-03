from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearRegression:

    l = 0
    theta = 0
    theta_rank = 14

    def __init__(self, l):
        self.l = l
        self.theta_rank = self.theta_rank
        self.theta = np.mat(np.zeros(self.theta_rank)).reshape((self.theta_rank, 1))

    def fit(self, x, y):
        '''
        get weight matrix
        :param x:
        :param y:
        :return: theta (14, 1)
        '''
        # x : (n, 14), type:matrix
        # y : (n, 1), type:matrix
        # n : the number of train data

        E = np.mat(np.identity(self.theta_rank))
        self.theta = (x.T*x + self.l*E).I * x.T * y

    def predict(self, x):
        '''
        predict
        :param x: n*13, type: matrix
        :return: y_pre, n*1 type: np.array
        '''
        y_pre = x * self.theta

        return np.array(y_pre)[:, 0]

if __name__ == '__main__':
    boston = load_boston()

    X = np.mat(boston.data).reshape((506, 13))
    y = np.mat(boston.target).reshape((506, 1))
    one = np.mat(np.ones(X.shape[0])).reshape((X.shape[0], 1))
    X = np.concatenate((X, one), axis=1)
    print("X shape is", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # print(type(x), '\n', type(y))
    # print(x.shape, '\n', y.shape)
    m = LinearRegression(0.1)
    m.fit(X_train, y_train)

    y_pre = m.predict(X_test)

    sse = np.array([i**2 for i in (y_pre - np.mean(y_test))]).sum()
    sst = np.array([i**2 for i in (y_test - np.mean(y_test))]).sum()
    ssr = np.array([i**2 for i in (np.array(y_test)[:, 0] - y_pre)]).sum()
    R_2 = sse / sst
    print("SSE = ", sse)
    print("SST = ", sst)
    print("SSR = ", ssr)
    print("R_2 = ", R_2)



    plt.plot(y_pre)
    plt.plot(np.array(y_test)[:, 0])
    plt.show()


