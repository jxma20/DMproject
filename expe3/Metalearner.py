import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class TreeNode:
    def __init__(self, feature=None, threshold=None, isleaf=False, label=None, lchild=None, rchild=None):
        self.feature = feature
        self.threshold = threshold
        self.isleaf = isleaf
        self.label = label
        self.lchild = lchild
        self.rchild = rchild

class MetaLearner(object):

    def __init__(self, min_samples, max_depth):
        self.head = None
        self.min_samples = min_samples
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        fit the model
        :param X: np.ndarray
        :param y: np.ndarray
        :return: None
        """
        y = y.reshape((len(y), 1))
        data = np.concatenate((X, y), axis=1)
        a = range(len(X[0]))
        # [0.0 1.0 2.0 3.0]
        self.head = self.TreeGenerate(data, a, 0)

    def predict(self, x):
        """
        get the predict result
        :param x: np.ndarray
        :return: np.ndarray
        """
        res = []
        for element in x:
            res.append(self.classifyone(element))

        return np.array(res, dtype=np.int64)

    def classifyone(self, single_x):
        """
        返回单个预测结果
        :param single_x:
        :return: int64
        """
        root = self.head
        while not root.isleaf:
            if single_x[root.feature] < root.threshold:
                root = root.lchild
            else:
                root = root.rchild
        return root.label

    def TreeGenerate(self, data, a, depth):
        """
        create Decision Tree
        :param data: dataset to be divided
        :param a: feature
        :return: Treenode
        """
        node = TreeNode()

        classList = [element[-1] for element in data]
        if len(set(classList)) == 1:
            return TreeNode(isleaf=True, label=classList[0])    #  返回classlist[0]

        count = sum([1 if i is not None else 0 for i in a])
        if count == 0 or depth == self.max_depth:
            return TreeNode(isleaf=True, label=(np.mean(data[:, -1])))   # 返回多数表决结果

        best_feature, best_threshold = self.FindBestFeature(data, a)

        node.feature = best_feature
        node.threshold = best_threshold

        L_data, R_data = self.split(data, best_feature, best_threshold)

        if len(L_data) == 0:
            node.lchild = TreeNode(isleaf=True, label=np.mean(data[:, -1]))
        else:
            node.lchild = self.TreeGenerate(L_data, a, depth+1)
        if len(R_data) == 0:
            node.rchild = TreeNode(isleaf=True, label=self.mean(data[:, -1]))
        else:
            node.rchild = self.TreeGenerate(R_data, a, depth+1)

        return node

    def FindBestFeature(self, data, a):
        """
        choose the feature with lowest gain
        :param data:
        :param a
        :return: best feature index
        """
        num_feature = len(a)
        best_feature = 0
        best_threshold = 0.0
        min_variance = math.inf
        for i in range(num_feature):
            # 寻找最佳切割点方差最小的feature为最佳feature
            # 找出feature的种类
            uniqueFeature = set([vec[i] for vec in data])
            # 对每个feature计算其对应的条件信息熵
            min_var = math.inf
            best_thre = 0.0
            for feature in uniqueFeature:
                # 以每一个feature为切割点，计算最好的切割点
                left_data, right_data = self.split(data, i, feature)
                if len(left_data) == 0:
                    l_variance = 0
                else:
                    l_variance = self.calc_variance(left_data)
                if len(right_data) == 0:
                    r_variance = 0
                else:
                    r_variance = self.calc_variance(right_data)
                variance = l_variance + r_variance
                if min_var > variance:
                    min_var = variance
                    best_thre = feature
            if min_var < min_variance:
                # 如果该属性的最小方差低于目前的最小方差，那么该属性和其分割阈值将被记录
                min_variance = min_var
                best_feature = i
                best_threshold = best_thre

        return best_feature, best_threshold

    def calc_variance(self, data):
        """
        以平均值为预测值，计算方差
        :param data:
        :return: 方差
        """
        labels = [vec[-1] for vec in data]
        aver = np.mean(labels)
        variance = sum([(i-aver) * (i-aver) for i in labels])
        return variance

    def mean(self, classes):
        return sum([i for i in classes])/len(classes)

    def split(self, data, axis, threshold):
        """
        按照axis的阈值将数据集分割为两个部分
        :param data:
        :param axis: 第几个特征
        :param threshold: 阈值
        :return: 数据集的两个部分
        """
        L, R = [], []
        for featVec in data:
            if featVec[axis] < threshold:
                L.append(featVec)
            else:
                R.append(featVec)
        return np.array(L), np.array(R)


    def ShowTree(self, root):
        """
        show my tree
        :param root:
        :return:
        """
        if root.isleaf:
            print("leaf: label is "+str(root.label))
            return
        print("node:  feature, "+str(root.feature)+" ", "threshold, " + str(root.threshold))
        self.ShowTree(root.lchild)
        self.ShowTree(root.rchild)

    def appraise_module(self, pre, y_test):
        return sum([(pre[i]-y_test[i])**2 for i in range(len(pre))])/len(pre)


if __name__ == "__main__":
    X, y = datasets.load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    print("------------init information-------------")
    print("the shape of X", X.shape)
    print("the shape of y", y.shape)
    print("the type of X", type(X))
    print("the type of y", type(y))

    print("---------------train log-----------------")

    dt = MetaLearner(5, 20)
    dt.fit(X_train, y_train)

    print("---------------show my tree--------------")
    dt.ShowTree(dt.head)

    print("---------------the result----------------")
    predict = dt.predict(X_test)
    print("predict type is", type(predict))
    variance = dt.appraise_module(predict, y_test)
    print("the variance is ", variance)