import math
def power(x, n):
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s

def ln(x, N=50):
    '''
    :param x: 输入值
    :param N: 迭代项数，缺省值为50
    :return: 对数值，误差的绝对值
    '''
    #   请在此添加实现代码   #
    # ********** Begin *********#
    sum=0
    for i in range(1,N+1):
        sum = sum +power(-1,i+1)*power(x-1,i)/i
    return  sum, math.fabs(math.log(x)-sum)





    # ********** End **********#