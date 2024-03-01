#coding=utf-8

from math import pi as PI

n =input()

# 请在此添加函数circle_area的代码，返回以n为半径的圆面积计算结果
#********** Begin *********#
def circle_area(n):
    if n.isdigit():
        N=int(n)
        s=PI*N*N
        return ('%.2f'%(s)) 
    else:
        return ('You must input an integer or float as radius.')
        # return None




#********** End **********#
print(circle_area(n))