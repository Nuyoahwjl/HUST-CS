#coding=utf-8

#输入数字字符串，并转换为数值列表
a = input()
num1 = eval(a)
numbers = list(num1)

# 请在此添加函数bubbleSort代码，实现编程要求
#********** Begin *********#
def bubbleSort(numbers):
    N=len(numbers)
    for i in range(1,N):
        for j in range(N-i):
            if numbers[j]>numbers[j+1]:
                temp=numbers[j]
                numbers[j]=numbers[j+1]
                numbers[j+1]=temp
    return numbers


#********** End **********#
print(bubbleSort(numbers))