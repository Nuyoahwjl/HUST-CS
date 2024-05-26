
list = ['python', 'matlab', 'java', 'c', 'c++']
count = 0
for book in list:
    count += 1
    if count == 3:
        continue
    print("当前书籍名字为:", book)
    if book == 'matlab':
        print(book, "可以节约算法验证的时间, 但只会", book, "容易被鄙视")