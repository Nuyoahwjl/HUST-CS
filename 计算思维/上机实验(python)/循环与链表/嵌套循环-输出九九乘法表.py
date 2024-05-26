 # 本程序要求打印九九乘法表前N行

N = int(input())

#   请在此添加实现代码   #
# ********** Begin *********#
for i in range(1,N+1):
    for j in range(1,i+1):
        if j<i:
            print(f"{i} * {j} = {j * i}", end="\t")
        else:
            print(f"{i} * {j} = {j * i}")
    # print()




# ********** End **********#