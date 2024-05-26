# para_tuple.py
# *numbers表示输入参数的个数可以是任意指
def myplus(*numbers):
    add = 0
    for i in numbers:
        add += i
    return add

# 调用3次plus函数, 每次参数个数都不相同

d1 = myplus(1,2,3)
d2 = myplus(1,2,3,4)
d3 = myplus(1,3,5,7,9)

# 向函数中可以传递0个参数
d4 = myplus()
print("d1=",d1,"d2=",d2,"d3=",d3,"d4=",d4)
