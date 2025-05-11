import numpy as np

def calcInfoGain(feature, label, index):
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
        # sub_label表示根据特征列和特征值分割出的子数据集中的标签
        sub_label = []
        for i in range(len(feature)):
            if feature[i][index] == value:
                count += 1
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


def calcInfoGainRatio(feature, label, index):
    '''
    计算信息增益率
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:信息增益率，类型float
    '''

    #********* Begin *********#
    total=label.size
    gain=calcInfoGain(feature, label, index)
    feature_col = feature[:, index]
    values, val_counts = np.unique(feature_col, return_counts=True)
    temp = 0.0
    for v, cnt in zip(values, val_counts):
        temp-=cnt/total*np.log2(cnt/total)
    return float(gain/temp)
    #********* End *********#