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

class SimpleDecisionTree(object):
    # def __init__(self, min_samples, min_gain, max_depth, max_leaves):
    #     self.min_samples = min_samples
    #     self.min_gain = min_gain
    #     self.max_depth = max_depth
    #     self.max_leaves = max_leaves
    #     self.head = TreeNode()
    def __init__(self):
        self.head = None

    def fit(self, data):
        """
        fit the model
        :param data:  np.ndarray
        :return: None
        """
        a = [float(i) for i in range(len(data[0])-1)]
        # [0.0 1.0 2.0 3.0]
        self.head = self.TreeGenerate(data, a)

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



    def TreeGenerate(self, data, a):
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
        if count == 0:
            return TreeNode(isleaf=True, label=(self.majority(data[:, -1])))   # 返回多数表决结果

        best_feature = self.FindBestFeature(data, a)
        # print("best_feature:", best_feature)
        a[best_feature] = None

        best_threshold = self.calc_best_threshold(data, best_feature)

        node.feature = best_feature
        node.threshold = best_threshold[0]

        L_data, R_data = self.split(data, best_feature, best_threshold)

        if len(L_data) == 0:
            node.lchild = TreeNode(isleaf=True, label=self.majority(data[:, -1]))
        else:
            node.lchild = self.TreeGenerate(L_data, a)
        if len(R_data) == 0:
            node.rchild = TreeNode(isleaf=True, label=self.majority(data[:, -1]))
        else:
            node.rchild = self.TreeGenerate(R_data, a)

        return node



    def FindBestFeature(self, data, a):
        """
        choose the feature with lowest gain
        :param data:
        :return: best feature index
        """
        num_feature = len(a)
        # 计算H(D)
        base_ent = self.calc_entropy(data)
        max_gain = 0.0
        best_feature = 0
        for i in range(num_feature):
            if a[i] is None:
                continue
            # 找出feature的种类
            uniqueFeature = set([vec[i] for vec in data])
            # 对每个feature计算其对应的条件信息熵
            sub_ent = 0.0
            for feature in uniqueFeature:
                sub_data = self.SplitDataSet(data, i, feature)
                # print("FindFeature: sub_data:", sub_data)
                prob = len(sub_data) / float(len(data))
                sub_ent += prob * self.calc_entropy(sub_data)
            # 计算每个feature对应的信息增益
            gain = base_ent - sub_ent
            # 选择最大的信息增益，其对应的feature即为最佳的feature
            if gain > max_gain:
                max_gain = gain
                best_feature = i
        return best_feature

    def majority(self, classes):
        """
        Majority vote
        :param classes:
        :return: the result of most numbers
        """
        classCount = {}
        for item in classes:
            if item in classCount.keys():
                classCount[item] += 1
            else:
                classCount[item] = 1
        sortedclassCount = sorted(classCount.items(), key=lambda d: d[1], reverse=True)
        return sortedclassCount[0][0]

    def calc_entropy(self, data):
        num = len(data)
        num_labels = {}
        # print("data calc_entropy shape is ", data.shape)
        for vec in data:
            if vec[-1] in num_labels.keys():
                num_labels[vec[-1]] += 1
            else:
                num_labels[vec[-1]] = 1
        ent = 0.0
        for key in num_labels.keys():
            prob = float(num_labels[key]) / num
            ent -= prob * math.log(prob, 2)
        return ent

    def calc_best_threshold(self, data, feature_index):
        """
        计算最优分割阈值
        :param data: 数据集
        :param feature_index: 某一属性的索引
        :return: float 分割阈值
        """
        best_gain = -math.inf
        best_threshold = None
        base_entropy = self.calc_entropy(data)
        for threshold in data[:, feature_index:feature_index+1]:
            L, R = self.split(data, feature_index, threshold)
            eve_entropy = 0.0
            for i in [L, R]:
                prob = len(i) / float(len(data))
                eve_entropy += prob*self.calc_entropy(i)
            gain = base_entropy - eve_entropy
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        return best_threshold




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

    def SplitDataSet(self, data, axis, feature_val):
        """
        区别于上面的split，该函数是根据相等是否跟某一属性值相等来分割
        :param data:
        :param axis:
        :param feature_val:
        :return:
        """
        retDataSet = []
        for featVec in data:
            if featVec[axis] == feature_val:
                retDataSet.append(featVec)
        return np.array(retDataSet)

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


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target.reshape((len(X), 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    print("------------init information-------------")
    print("the shape of X", X.shape)
    print("the shape of y", y.shape)
    print("the type of X", type(X))
    print("the type of y", type(y))

    data = np.concatenate((X_train, y_train), axis=1)
    print("the shape of data", data.shape)
    print("data:", data[:5])

    print("---------------train log-----------------")

    dt = SimpleDecisionTree()
    dt.fit(data)

    print("---------------show my tree--------------")
    dt.ShowTree(dt.head)

    print("---------------the result----------------")
    predict = dt.predict(X_test)
    corr_rate = (sum([y_test[i] == predict[i] for i in range(len(y_test))]) / len(y_test))[0]
    print("the rate of correction is", corr_rate)
