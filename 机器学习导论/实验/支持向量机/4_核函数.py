#encoding=utf8

import numpy as np

#实现核函数
def kernel(x,sigma=1.0):
    '''
    input:x(ndarray):样本
    output:x(ndarray):转化后的值
    '''    
    #********* Begin *********#
    # 计算样本间的平方欧氏距离矩阵
    x_sq = np.sum(x**2, axis=1)
    dist_sq = x_sq[:, np.newaxis] + x_sq[np.newaxis, :] - 2 * np.dot(x, x.T)
    # 应用高斯核公式
    gamma = 1.0 / (2 * sigma**2)
    k = np.exp(-gamma * dist_sq)
    return k
    #********* End *********#
