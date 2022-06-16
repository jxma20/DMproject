import dm2022exp
import numpy as np
from collections import Counter

class KMeans:
    def __init__(self, num_cluster, max_iter=1000, tol=1e-4):
        self.num_cluster = num_cluster
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """
        对数据进行分类
        :param X: 待分类的样本数据
        :return: np.ndarray 分类的结果
        """
        len_x = len(X)
        random_index = np.random.randint(len_x, size=(self.num_cluster, ))
        cluster = [X[i] for i in random_index]  # (8, 2)
        for epoch in range(self.max_iter):
            print(epoch)
            res = self.result_from_cluster(cluster, X)  # 分类的结果
            # 重新求每个簇的中心点，得到新的候选点
            new_cluster = self.get_new_cluster(res, X)
            if self.distance_now_last(cluster, new_cluster) < self.tol:
                return self.result_from_cluster(new_cluster, X)
            cluster = new_cluster
        return self.result_from_cluster(cluster, X)

    def get_new_cluster(self, result, X):
        """
        通过分类结果得出新的候选点
        :param result: (4000,)
        :param X: (4000, 2)
        :return: np.ndarray (8, 2)
        """
        new_cluster = []
        for i in range(self.num_cluster):
            # 对i类的数据进行提取,保存其在X中的下标
            clus = [j for j in range(len(result)) if result[j] == i]
            # 取出分类的数据
            X_split = np.array([X[j] for j in clus])
            # 按列求平均
            one_cluster = np.mean(X_split, axis=0)
            new_cluster.append(one_cluster)
        return new_cluster

    def distance_now_last(self, last, now):
        """
        返回这一次候选点与上一次候选点的距离均值
        :param last:
        :param now:
        :return: float
        """
        l = len(last)
        dis = []
        for index in range(l):
            dis.append(self.calc_dis_two_point(last[index], now[index]))
        distance = sum(dis)
        return distance

    def calc_dis_two_point(self, a, b):
        """
        计算a, b两点的距离
        :param a: 2-D ,坐标(a[0], a[1])
        :param b: 2-D ,坐标(b[0], b[1])
        :return: float , 距离
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def result_from_cluster(self, cluster, X):
        """
        通过对每个点求它最近的候选点判断该点属于的簇，得到所有的簇
        :param cluster:
        :param X:
        :return: np.ndarray (4000,)
        """
        result = np.zeros(len(X))
        for i in range(len(X)):
            clus = 0
            dis = np.inf
            for j in range(len(cluster)):
                d = self.calc_dis_two_point(X[i], cluster[j])
                if d < dis:
                    dis = d
                    clus = j
            result[i] = clus
        return result

    def purity(self, y_pred, y):
        """
        参数解释:
            - y_pred  : 聚类的预测值
            - y       : 真实的类别
            - n_class : 一共 `n_class` 个类别
        返回值：预测值 `y_pred` 的纯度
        """
        pr = 0
        for i in range(self.num_cluster):
            idxs = y_pred == i
            cnt = Counter(y[idxs])
            pr += cnt.most_common()[0][1]
        return pr / len(y)



if __name__ == "__main__":
    X, y = dm2022exp.load_ex4_data()
    print("------------the data information-----------")
    print("X shape is :", X.shape)
    print("y shape is :", y.shape)
    print("X的前五个数据\n", X[:5, :])

    # dm2022exp.show_exp4_data(X, y)

    print("---------------the result------------------")
    km = KMeans(8, 1000)
    result = km.fit(X)
    print("the result is :\n", result)
    dm2022exp.show_exp4_data(X, result)
    print("the purity is:", km.purity(result, y))
