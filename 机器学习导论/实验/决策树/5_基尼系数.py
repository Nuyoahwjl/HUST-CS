import numpy as np

def calcGini(feature, label, index):
    '''
    计算基尼系数
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:基尼系数，类型float
    '''

    #********* Begin *********#
    total=label.size
    feature_col=feature[:, index]
    values, val_counts=np.unique(feature_col, return_counts=True)
    gini=0.0
    for v, cnt in zip(values, val_counts):
        sub_labels=label[feature_col == v]
        sub_total=cnt
        sub_labels_counts = np.unique(sub_labels, return_counts=True)[1]
        sub_gini=1.0
        for sub_cnt in sub_labels_counts:
            p_sub = sub_cnt / sub_total
            sub_gini-=p_sub**2
        gini += (cnt / total) * sub_gini
    return gini
    #********* End *********#