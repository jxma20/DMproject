import dm2022exp
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

class Apriori:
    le = None
    freq_sup = {}

    def __init__(self, min_sup=0.2, min_conf=0.5):
        self.min_sup = min_sup
        self.min_conf = min_conf

    def fit(self, X):
        """
        返回数据集X中所有支持度大于min_sup的频繁项集
        :param X:X中有多个列表，每个列表中包含多个字符串（事务）
        :return:返回的数据为一串列表，其中每个列表中包含两项，第一项是frozenset类型，它里面包含一个频繁项集，第二项是该频繁项集的支持度。
        """
        data = self.process_data(X)
        num = len(self.le.classes_)
        print("商品种类数：", num)
        # 首先得到f1
        list = [[i] for i in range(num)]
        # [print(i) for i in data]
        f1 = [(i, self.calc_sup(frozenset(i), data)) for i in list if self.calc_sup(frozenset(i), data) >= self.min_sup]
        f1_copy = [(frozenset(self.le.inverse_transform(element[0])), element[1]) for element in f1]
        f1 = [(frozenset(element[0]), element[1]) for element in f1]
        print("f1 length is :", len(f1))
        '''
        整体流程：
        根据f1得到f2的候选->get_candidates()，
        然后计算f2中项集每一项的支持度->calc_sup()，
        将其中支持度大于min_sup的挑选出来组成f2，
        将此作为一个循环
        '''
        freq_set = f1
        relation_rule_dict = {}
        res = f1_copy
        while True:
            new_candidate = self.get_candidates(freq_set)
            if len(new_candidate) == 0:
                break

            freq_set = [(i, self.calc_sup(i, data)) for i in new_candidate if self.calc_sup(i, data) >= self.min_sup]
            self.UpdateRelationRule(freq_set, relation_rule_dict)
            print("new_frequuent_set lenth is:", len(freq_set))
            print("the first epoch is finished")

            freq_set_copy = [(frozenset(self.le.inverse_transform([j for j in i[0]])), i[1]) for i in freq_set]
            [res.append(i) for i in freq_set_copy]
        return res, relation_rule_dict

    def UpdateRelationRule(self, freq_set_k, res_dict_relation):
        """
        更新保存关联规则的字典
        :param freq_set_k: 某一k频繁项集
        :param res_dict_relation: 保存关联规则的字典
        :return:
        """
        for value in freq_set_k:
            element_list = self.get_sub_set(list(value[0]))
            element_support = self.freq_sup[value[0]]
            for number in element_list:
                number_set = frozenset(number)
                no_number_set = value[0].difference(number_set)
                number_set_support = self.freq_sup[number_set]
                confidence = element_support / number_set_support
                if confidence >= self.min_conf:
                    temp1_set = set([i for i in self.le.inverse_transform([i for i in number_set])])
                    temp2_set = set([i for i in self.le.inverse_transform([i for i in no_number_set])])
                    res_dict_relation[str(temp1_set) + '->' + str(temp2_set)] = confidence

    def get_sub_set(self, nums):
        """
        给定一个列表，返回一个含有该列表所有子集的列表
        :param nums:
        :return:
        """
        sub_sets = [[]]
        for x in nums:
            sub_sets.extend([item + [x] for item in sub_sets])
            pass
        sub_sets.remove([])
        sub_sets.remove(nums)
        return sub_sets

    def get_candidates(self, formal):
        """
        通过上一个频繁项集得到下一个频繁项集
        :param formal:
        :return:
        """
        res = []
        temp_set = set()
        for i in range(len(formal)):
            for j in range(i+1, len(formal)):
                if self.is_same_prefix(formal[i][0], formal[j][0]):
                    t = formal[i][0] | formal[j][0]
                    if t in temp_set:
                        continue
                    res.append(t)
                    temp_set.add(t)
        return res

    def is_same_prefix(self, x, y):
        """
        判断x和y的前n-1项是否相等
        :param x:
        :param y:
        :return:
        """
        # for index in range(len(x)-1):
        #     if x[index] != y[index]:
        #         return False
        if len(x & y) == len(x) - 1:
            return True
        return False

    def calc_sup(self, x, data):
        """
        计算x在data中的支持度
        :param x: list[1,2,..]
        :param data: 数据集
        :return: float
        """
        if x in self.freq_sup.keys():
            return self.freq_sup[x]
        count = 0
        for affair in data:
            if x.issubset(affair):
                count += 1
        # print(count/len(data))
        self.freq_sup[x] = count/len(data)
        return count/len(data)



    def process_data(self, data):
        """
        利用编码器将data中以字符串构成的事务转换为数字
        :param data: 源数据
        :return: list[np.array, np.array,..., np.array]
        """
        self.le = LabelEncoder()
        list_1d = []
        for i in data:
            for j in i:
                list_1d.append(j)
        self.le.fit(list_1d)
        return [set(self.le.transform(i)) for i in data]

if __name__ == "__main__":
    data = dm2022exp.load_ex5_data()
    print("data type is:", type(data))
    [print(i) for i in data[:5]]
    print("-------------------------------the train process-----------------------------")
    ap = Apriori(0.005, 0.55)
    res, rela_rule = ap.fit(data)
    print("--------------------------------my result show time-------------------------")
    print("the res is:------------------------------------------------------------------")
    print("the res length is:", len(res))
    [print(i) for i in res]
    print("the relation is :------------------------------------------------------------")
    print("the length of relation rule is :", len(rela_rule))
    [print(key, ":", rela_rule[key]) for key in rela_rule]

    print("-------------------------------mlxtend result is following--------------------------------")
    m = TransactionEncoder()
    m.fit(data)
    m.transform(data)
    df = pd.DataFrame(m.transform(data), columns=m.columns_)
    ret = apriori(df, min_support=0.005, use_colnames=True, verbose=True)
    rule = association_rules(ret, metric="confidence", min_threshold=0.55)
    print(ret)
    print(rule['antecedents'])
    print(rule['consequents'])
    print(rule['confidence'])