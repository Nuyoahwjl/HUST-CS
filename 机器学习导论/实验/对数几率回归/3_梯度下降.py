# -*- coding: utf-8 -*-

import numpy as np
import warnings
warnings.filterwarnings("ignore")

def gradient_descent(initial_theta,eta=0.05,n_iters=1000,epslion=1e-8):
    '''
    梯度下降
    :param initial_theta: 参数初始值，类型为float
    :param eta: 学习率，类型为float
    :param n_iters: 训练轮数，类型为int
    :param epslion: 容忍误差范围，类型为float
    :return: 训练后得到的参数
    '''
    #   请在此添加实现代码   #
    #********** Begin *********#
    theta = initial_theta
    for _ in range(n_iters):
        grad = 2 * (theta - 3) # 计算梯度
        if abs(grad) < epslion: # 检查梯度是否足够小
            break
        theta -= eta * grad # 更新参数
    return theta
    #********** End **********#
