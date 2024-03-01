from math import *

def print_(x):
    if type(x) == float:
        print("%.4f" % x)
    else:
        print(x)
# ********** Begin ********** #
#第一题
for temperature in [-271, -100, 0, 100, 1000]:
    #请在下面编写代码
    F=9/5*temperature+32
    #请不要修改下面的代码
    print_(F)

print('\n***********************\n')

#第二题
for (m, s, x) in [(0,2,1),(1,2,2),(1,3,4),(1,10,100)]:
    # 请在下面编写代码
    fx=(1/sqrt(2*pi*s))*(e**(0-(((x-m)/s)**2)/2))
    # 请不要修改下面的代码
    print_(fx)

print('\n***********************\n')

#第三题
for x in [0.0, pi/2, pi, 3*pi/2, 2*pi, 5*pi/2, 3*pi]:
    # 请在下面编写代码
    sinh=(exp(x)-exp(0-x))/2
    # 请不要修改下面的代码
    print_(sinh)


print('\n***********************\n')

#第四题
g = 9.8
for v0 in [10, 15, 20, 25, 30]:
    for t in [0.0, 0.5, 1, 1.5, 2, 2.5, 3]:
        # 请在下面编写代码
        y=v0*t-(g*t*t)/2
        # 请不要修改下面的代码
        print_(y)
    print('***********************')
# ********** End ********** #