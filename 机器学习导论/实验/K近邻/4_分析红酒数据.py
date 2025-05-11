import numpy as np
from sklearn.datasets import load_wine
wine_dataset = load_wine()

"""
# 打印红酒数据集中的特征的名称
print(wine_dataset['feature_names'])
# 打印红酒数据集中的标签的名称
print(wine_dataset['target_names'])
# 查看数据集内容
print("特征数据形状:", wine_dataset.data.shape)  # (178, 13)
print("标签数据形状:", wine_dataset.target.shape)  # (178,)
print("特征名称:", wine_dataset.feature_names)
print("类别名称:", wine_dataset.target_names)
print("第一个样本的特征值:", wine_dataset.data[0])
print("第一个样本的标签:", wine_dataset.target[0])
# 打印数据集描述
print(wine_dataset.DESCR)
"""

def alcohol_mean(data):
    '''
    返回红酒数据中红酒的酒精平均含量
    :param data: 红酒数据对象
    :return: 酒精平均含量 类型为float
    '''
    
    #********* Begin *********#
    # 获取特征数据和特征名称
    d = data.data  # 特征数据 (178, 13)
    feature_names = data.feature_names  # 特征名称列表
    # 找到酒精含量对应的列索引
    alcohol_index = feature_names.index('alcohol')
    # 提取酒精含量列
    alcohol_values = d[:, alcohol_index]
    # 计算酒精含量的平均值
    average_alcohol = alcohol_values.mean()
    return average_alcohol
    #********* End **********#