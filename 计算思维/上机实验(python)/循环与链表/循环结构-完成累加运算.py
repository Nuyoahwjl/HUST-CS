 # 本程序计算1-N整数平方的累加和

N = int(input())

#   请在此添加实现代码   #
# ********** Begin *********#
sum = 0
for x in range(N+1):
    sum = sum + x*x
print(sum)



# ********** End **********#
