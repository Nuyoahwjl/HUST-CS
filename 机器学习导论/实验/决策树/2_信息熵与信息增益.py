import numpy as np


def calcInfoGain(feature, label, index):
    '''
    计算信息增益
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:信息增益，类型float
    '''

    #*********** Begin ***********#
    # 计算原始熵
    total = label.size
    if total == 0:
        return 0.0
    labels, label_counts = np.unique(label, return_counts=True)
    original_entropy = 0.0
    for count in label_counts:
        p = count / total
        original_entropy -= p * np.log2(p)
    
    # 计算条件熵
    feature_col = feature[:, index]
    values, val_counts = np.unique(feature_col, return_counts=True)
    conditional_entropy = 0.0
    for v, cnt in zip(values, val_counts):
        sub_labels = label[feature_col == v]
        sub_total = cnt
        sub_labels_counts = np.unique(sub_labels, return_counts=True)[1]
        sub_entropy = 0.0
        for sub_cnt in sub_labels_counts:
            p_sub = sub_cnt / sub_total
            sub_entropy -= p_sub * np.log2(p_sub)
        conditional_entropy += (cnt / total) * sub_entropy
    
    info_gain = original_entropy - conditional_entropy
    return float(info_gain)
    #*********** End *************#