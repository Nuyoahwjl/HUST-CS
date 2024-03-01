from math import *

def print_(x):
    if type(x) == float:
        print("%.4f" % x)
    else:
        print(x)
# ********** Begin ********** #
#请在每一题的print语句内完成题目所需的表达式

#第一题
print_(pi**4+pi**5)
print_(e**6)
print_(pi**4+pi**5-e**6)

#第二题
print_(pi/4)
print_(4*atan(1/5)-atan(1/239))

#第三题
print_(cos(2*pi/17))
print_(((-1)+sqrt(17)+sqrt(2*(17-sqrt(17)))+2*sqrt(17+3*sqrt(17)-sqrt(2*(17-sqrt(17)))-2*sqrt(2*(17+sqrt(17)))))/16)
print_(cos(2*pi/17)-(((-1)+sqrt(17)+sqrt(2*(17-sqrt(17)))+2*sqrt(17+3*sqrt(17)-sqrt(2*(17-sqrt(17)))-2*sqrt(2*(17+sqrt(17)))))/16))

#第四题
print_(sqrt((1+sqrt(5))/2+2)-(1+sqrt(5))/2)

#第五题
print_(sinh(0.25))
print_(((e**0.25)-(e**(-0.25)))/2)

# ********** End ********** #