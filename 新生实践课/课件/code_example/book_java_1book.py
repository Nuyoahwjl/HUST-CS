
list = ['python', 'matlab', 'java', 'c', 'c++']
count = 0
for 1book in list: # 标识符以数字开头所以报错 无法运行
    count += 1
    if count == 3:
        continue
    print("当前书籍名字为:", 1book)
    if 1book == 'matlab':
        print(1book, "可以节约算法验证的时间, 但只会", book, "容易被鄙视")