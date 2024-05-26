# para_dict.py
#把赋值的等号两边变成key和valus
def myplus(**number):
    return number

d1 = myplus() # 向函数中可以传递0个参数
d2 = myplus(x=1)
d3 = myplus(x1=1, y1=2)

print("d1=",d1,"d2=",d2,"d3=",d3)
