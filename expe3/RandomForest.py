import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn import datasets
import Metalearner as ml # 这是我自己实现的回归决策树类
import numpy as np
import matplotlib.pyplot as plt

class Random_Frorest:
    def __init__(self, n_tree):
        self.N_tree = n_tree
        self.trees = [ml.MetaLearner(5, 20) for i in range(n_tree)]
        self.X_test = None
        self.y_test = None

    def fit(self, X, y):
        """
        对每一棵树进行训练
        :param X:
        :param y:
        :return:
        """
        data = self.process_X_y_to_data(X, y)
        for index_tree in range(len(data)):
            self.trees[index_tree].fit(data[index_tree][:, :len(X[0])], data[index_tree][:, -1])

    def predict(self):
        """
        使用得到的X_test和y_test分别输入到每一棵树中，然后再对得出的结果进行求平均值
        :return: np.ndarray
        """
        res = []
        for tree in self.trees:
            p = tree.predict(self.X_test)
            res.append(p)
        return np.mean(np.array(res), 0)

    def process_X_y_to_data(self, x, y):
        """
        将X，y合称为一个data，然后对其进行有放回抽样，每次抽样数量为N=len(X)次，这是一个样本，总共需要N_tress个样本
        :param x: np.ndarray
        :param y: np.ndarray
        :return:  np.ndarray, 三维数据，由N_trees个data组成
        """
        y = y.reshape((len(y), 1))
        d = np.concatenate((x, y), axis=1)
        N = len(d)
        data = []
        choosed = np.zeros(N)
        for i in range(self.N_tree):
            random_d = []
            for j in range(N):
                randIndex = np.random.randint(0, N)
                random_d.append(d[randIndex])
                choosed[randIndex] = 1
            data.append(random_d)

        self.X_test = np.array([x[i] for i in range(N) if choosed[i] == 0])
        self.y_test = np.array([y[i] for i in range(N) if choosed[i] == 0])
        return np.array(data)

    def calc_square(self, predict):
        return sum([(predict[i]-self.y_test[i])**2 for i in range(len(predict))])/len(predict)

if __name__ == "__main__":
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    # t = Random_Frorest(12)
    # print("------------my randomforest-----------------")
    # t.fit(X_train, y_train)
    # t.X_test = X_test
    # t.y_test = y_test
    # predict = t.predict()
    # print("predict shape is :", predict.shape)
    # print("my forest MSE is :", t.calc_square(predict))

    # print("------------use sklearn--------------------")
    # skl_rf = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
    # skl_rf.fit(X_train, y_train)
    # predict_skl = skl_rf.predict(X_test)
    # print("sklearn MSE is :", t.calc_square(predict_skl))

    print("--------------task 3-----------------")
    x = np.linspace(1, 30, 30, dtype=np.int64)
    mse = []
    for i in x:
        print(i)
        rf = Random_Frorest(i)
        rf.fit(X_train, y_train)
        rf.X_test = X_test
        rf.y_test = y_test
        mse.append(rf.calc_square(rf.predict()))
    print(mse)
    plt.plot(mse)
    plt.xlabel("the number of trees")
    plt.ylabel("MSE")

    plt.show()


