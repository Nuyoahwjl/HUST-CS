# range_example.py
print(range(10)) # 默认从0开始
print(range(0, 10))
numbers = []
for i in range(10):
    number = i**2 # 请问这句话的意思?
    numbers.append(number) # 用append追加列表的新元素

print(numbers)

a = tuple(numbers)
# a[0] = 5 # 打开注释就报错, 因为元组不允许二次赋值
print(a)