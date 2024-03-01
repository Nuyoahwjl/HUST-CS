# 请用函数实现arctg泰勒级数计算，包含隐含参数N
def power(x, n):
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s

def arctg(x, N=5):
    '''
    :param x: 输入值
    :param N: 迭代项数，缺省值为5
    :return: arctg值
    '''
    #   请在此添加实现代码   #
    # ********** Begin *********#
    sum=0
    for i in range(1,N+1):
        sum = sum +power(-1,i-1)*power(x,2*i-1)/(2*i-1)
    return sum

    
    # ********** End **********#