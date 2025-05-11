import numpy as np

class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        例如训练数据集中的特征为 [[2, 1, 1],
                              [1, 2, 2],
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.condition_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}

    def fit(self, feature, label):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        '''

        #********* Begin *********#

        # 计算各类别的先验概率
        # 返回label中有几个不重复的值以及出现的次数
        unique_labels, label_counts = np.unique(label, return_counts=True)
        total_samples = len(label)
        self.label_prob = {label: count / total_samples for label, count in zip(unique_labels, label_counts)}
        
        # 确定每个特征列的所有可能取值
        n_features = feature.shape[1]
        unique_values_per_col = [np.unique(feature[:, j]) for j in range(n_features)]
        
        # 初始化条件概率结构
        self.condition_prob = {}
        for label_val in unique_labels:
            self.condition_prob[label_val] = {}
            # 获取当前类别的样本
            mask = (label == label_val)
            samples_in_label = feature[mask]
            # 遍历每个特征列
            for j in range(n_features):
                self.condition_prob[label_val][j] = {}
                # 当前特征列的可能取值
                possible_values = unique_values_per_col[j]
                possible_values_len = len(possible_values)
                # 当前列的数据
                col_data = samples_in_label[:, j]
                total = len(col_data)
                # 统计每个值的出现次数并计算概率
                for value in possible_values:
                    count = np.sum(col_data == value)
                    prob = (count+1) / (total+possible_values_len)
                    self.condition_prob[label_val][j][value] = prob
                    
        #********** End **********#


    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''

        result = []
        # 对每条测试数据都进行预测
        for i, f in enumerate(feature):
            # 可能的类别的概率
            prob = np.zeros(len(self.label_prob.keys()))
            ii = 0
            for label, label_prob in self.label_prob.items():
                # 计算概率
                prob[ii] = label_prob
                for j in range(len(feature[0])):
                    prob[ii] *= self.condition_prob[label][j][f[j]]
                ii += 1
            # 取概率最大的类别作为结果
            result.append(list(self.label_prob.keys())[np.argmax(prob)])
        return np.array(result)