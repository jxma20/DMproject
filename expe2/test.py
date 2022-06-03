from math import *
import operator
import pickle


# 计算熵
def entropy(data):
    num = len(data)
    num_labels = {}
    for vec in data:
        num_labels[vec[-1]] = num_labels.get(vec[-1], 0) + 1
    ent = 0.0
    for key in num_labels.keys():
        prob = float(num_labels[key]) / num
        ent -= prob * log(prob, 2)
    return ent


# 根据给定特征划分数据集, axis是特征对应的编号，value是匹配的值
def splitDataSet(data, axis, value):
    res = []
    for vec in data:
        if vec[axis] == value:
            res.append(vec[:axis] + vec[axis + 1:])
    return res


# 遍历数据集，计算每种特征对应的信息增益，选出增益最小的，该特征即为用于分类的特征
def chooseFeature(data):
    num_feature = len(data[0]) - 1
    # 计算H(D)
    base_ent = entropy(data)
    max_gain = 0.0
    best_feature = 0
    for i in range(num_feature):
        # 找出feature的种类
        uniqueFeature = set([vec[i] for vec in data])
        # 对每个feature计算其对应的条件信息熵
        sub_ent = 0.0
        for feature in uniqueFeature:
            sub_data = splitDataSet(data, i, feature)
            prob = len(sub_data) / float(len(data))
            sub_ent += prob * entropy(sub_data)
        # 计算每个feature对应的信息增益
        gain = base_ent - sub_ent
        # 选择最大的信息增益，其对应的feature即为最佳的feature
        if gain > max_gain:
            max_gain = gain
            best_feature = i
    return best_feature


# 递归构建决策树
# 原始数据集-> 递归地基于最好的属性划分数据集->终止条件：所有属性已经用过或划分后的数据集每个集合只属于一个label,
# 若所有属性用完但是每个划分属性不都属于一个label，就使用“多数表决”的方法。

# 多数表决函数
def majority(classes):
    classCount = {}
    for item in classes:
        classCount[item] = classCount.get(item, 0) + 1
    sortedClassesCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回表决最多的label
    return sortedClassesCount[0][0]


# 递归构建树，存入字典中，以便随后的可视化.feature_name是feature的名字，仅是为了可视化方便。
def createTree(data, feature_names):
    # 找出data数据集中所有的labels
    classList = [item[-1] for item in data]
    # 如果只属于一种label，则停止构建树
    if len(set(classList)) == 1:
        return classList[0]
    # 若果所有features属性都已经使用完，也停止构建树
    # 每次用完一个特征都会删除，这样最后数据集data中只剩下最后一位的label位
    if len(data[0]) == 1:
        return majority(classList)
    bestFeat = chooseFeature(data)
    bestFeatName = feature_names[bestFeat]
    # bestFeatName是用于分离数据集的特征的名字，作为根
    tree = {bestFeatName: {}}
    # del只删除元素，但是原来的index不变
    sub_names = feature_names[:]
    del (sub_names[bestFeat])
    uniqFeats = set([item[bestFeat] for item in data])
    # 对于每个feature，递归地构建树
    for feature in uniqFeats:
        tree[bestFeatName][feature] = createTree(splitDataSet(data, bestFeat, feature), sub_names)
    return tree


# 分类函数,也是递归地分类（从根到叶节点）
def classify(tree, feature_names, test_data):
    # 找到根，即第一个用于分类的特征
    root = list(tree.keys())[0]
    # 找到根对应的子树
    sub_trees = tree[root]
    # 找出对应该特征的index
    feat_index = feature_names.index(root)
    # 对所有子树，将测试样本划分到属于其的子树
    for key in sub_trees.keys():
        if test_data[feat_index] == key:
            # 检查是否还有子树，或者已经到达叶节点
            if type(sub_trees[key]).__name__ == 'dict':
                # 若还可以再分，则继续向下找
                return classify(sub_trees[key], feature_names, test_data)
            else:
                # 否则直接返回分的类
                return sub_trees[key]


# 存储树（存储模型）
def stroeTree(tree, filename):
    fw = open(filename, 'w')
    pickle.dump(tree, fw)
    fw.close()
    return


# 提取树
def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)


# 预测隐形眼镜类型，已知数据集，如何判断患者需要佩戴的眼镜类型
fr = open('lenses.txt')
lenses = [line.strip().split('\t') for line in fr.readlines()]
feature_names = ['age', 'prescript', 'astigmatic', 'tearRate']
# 构建树
tree = createTree(lenses, feature_names)
# 可视化树
# createPlot(tree)
# 测试
print(classify(tree, feature_names, ['pre', 'myope', 'no', 'reduced']))