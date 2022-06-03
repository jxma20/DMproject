from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import tree
from matplotlib import pyplot as plt
from IPython.display import Image
import pydotplus

def load_data():
    iris = datasets.load_iris() # scikit-learn 自带的 iris 数据集
    X_train = iris.data
    y_train = iris.target
    return train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


X_train, X_test, y_train, y_test = load_data()
score = [0.98]
criterions=['gini', 'entropy']
i = 1
for criterion in criterions:
    clf = tree.DecisionTreeClassifier(criterion=criterion)
    clf.fit(X_train, y_train)
    # print(criterion, "Training score:%f" % (clf.score(X_train, y_train)))
    s = clf.score(X_test, y_test)
    print(criterion, "Testing corrected rate:%f" % s)
    score.append(s)
    # regr_1 和regr_2
    dot_data = tree.export_graphviz(clf, out_file=None,  # regr_1 是对应分类器
                                    feature_names=['f1', 'f2', 'f3', 'f4'],  # 对应特征的名字
                                    class_names=['0', '1', '2'],  # 对应类别的名字
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('example'+str(i)+".png")  # 保存图像
    i += 1
    Image(graph.create_png())

name = ["ID3", "CART", "C4.5"]
plt.bar(name, score)
plt.title("the corrected rate of distinct method")
plt.show()


