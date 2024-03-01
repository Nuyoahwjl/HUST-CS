#!/usr/bin/python
# -*- coding: UTF-8 -*-

count=0
def move(n, a, b, c):
    global count
    if n==1:
   # 请在此添加代码，勿改动其它代码
    #-----------Begin----------
        print(a,'-->',c)
        count=count+1
        return

    #------------End-----------
    else:
    # 请在此添加代码，勿改动其它代码
    #-----------Begin----------
        move(n-1,a,c,b)  
        move(1,a,b,c)   
        move(n-1,b,a,c)  

    #------------End-----------       
        
n=int(input('请输入盘子的个数：'))
move(n, 'A', 'B', 'C')
print("总共移动的次数：",count)
