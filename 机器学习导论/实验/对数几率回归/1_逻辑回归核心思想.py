#encoding=utf8
import numpy as np

def sigmoid(t):
    '''
    完成sigmoid函数计算
    :param t: 负无穷到正无穷的实数
    :return: 转换后的概率值
    :可以考虑使用np.exp()函数
    '''
    #********** Begin **********#
    return 1/(1+np.exp(-t))
    #********** End **********#