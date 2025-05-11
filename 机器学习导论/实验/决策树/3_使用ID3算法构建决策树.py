import numpy as np
class DecisionTree(object):
    def __init__(self):
        #决策树模型
        self.tree = {}
    def calcInfoGain(self, feature, label, index):
        '''
        计算信息增益
        :param feature:测试用例中字典里的feature，类型为ndarray
        :param label:测试用例中字典里的label，类型为ndarray
        :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
        :return:信息增益，类型float
        '''
        # 计算熵
        def calcInfoEntropy(label):
            '''
            计算信息熵
            :param label:数据集中的标签，类型为ndarray
            :return:信息熵，类型float
            '''
            label_set = set(label)
            result = 0
            for l in label_set:
                count = 0
                for j in range(len(label)):
                    if label[j] == l:
                        count += 1
                # 计算标签在数据集中出现的概率
                p = count / len(label)
                # 计算熵
                result -= p * np.log2(p)
            return result
        # 计算条件熵
        def calcHDA(feature, label, index, value):
            '''
            计算信息熵
            :param feature:数据集中的特征，类型为ndarray
            :param label:数据集中的标签，类型为ndarray
            :param index:需要使用的特征列索引，类型为int
            :param value:index所表示的特征列中需要考察的特征值，类型为int
            :return:信息熵，类型float
            '''
            count = 0
            # sub_feature和sub_label表示根据特征列和特征值分割出的子数据集中的特征和标签
            sub_feature = []
            sub_label = []
            for i in range(len(feature)):
                if feature[i][index] == value:
                    count += 1
                    sub_feature.append(feature[i])
                    sub_label.append(label[i])
            pHA = count / len(feature)
            e = calcInfoEntropy(sub_label)
            return pHA * e
        base_e = calcInfoEntropy(label)
        f = np.array(feature)
        # 得到指定特征列的值的集合
        f_set = set(f[:, index])
        sum_HDA = 0
        # 计算条件熵
        for value in f_set:
            sum_HDA += calcHDA(feature, label, index, value)
        # 计算信息增益
        return base_e - sum_HDA
    # 获得信息增益最高的特征
    def getBestFeature(self, feature, label):
        max_infogain = 0
        best_feature = 0
        for i in range(len(feature[0])):
            infogain = self.calcInfoGain(feature, label, i)
            if infogain > max_infogain:
                max_infogain = infogain
                best_feature = i
        return best_feature
    def createTree(self, feature, label):
        # 样本里都是同一个label没必要继续分叉了
        if len(set(label)) == 1:
            return label[0]
        # 样本中只有一个特征或者所有样本的特征都一样的话就看哪个label的票数高
        if len(feature[0]) == 1 or len(np.unique(feature, axis=0)) == 1:
            vote = {}
            for l in label:
                if l in vote.keys():
                    vote[l] += 1
                else:
                    vote[l] = 1
            max_count = 0
            vote_label = None
            for k, v in vote.items():
                if v > max_count:
                    max_count = v
                    vote_label = k
            return vote_label
        # 根据信息增益拿到特征的索引
        best_feature = self.getBestFeature(feature, label)
        tree = {best_feature: {}}
        f = np.array(feature)
        # 拿到bestfeature的所有特征值
        f_set = set(f[:, best_feature])
        # 构建对应特征值的子样本集sub_feature, sub_label
        for v in f_set:
            sub_feature = []
            sub_label = []
            for i in range(len(feature)):
                if feature[i][best_feature] == v:
                    sub_feature.append(feature[i])
                    sub_label.append(label[i])
            # 递归构建决策树
            tree[best_feature][v] = self.createTree(sub_feature, sub_label)
        return tree
    def fit(self, feature, label):
        '''
        :param feature: 训练集数据，类型为ndarray
        :param label:训练集标签，类型为ndarray
        :return: None
        '''
        #************* Begin ************#
        self.tree=self.createTree(feature, label)
        #************* End **************#
    def predict(self, feature):
        '''
        :param feature:测试集数据，类型为ndarray
        :return:预测结果，如np.array([0, 1, 2, 2, 1, 0])
        '''
        #************* Begin ************#
        result = []
        for sample in feature:
            node = self.tree
            while isinstance(node, dict):
                # 获取当前分裂特征索引
                feature_idx = list(node.keys())[0]
                # 根据特征值选择子树分支
                value = sample[feature_idx]
                node = node[feature_idx][value]
            result.append(node)
        return np.array(result)
        #************* End **************#