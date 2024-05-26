# sayhello.py
def printsayhello(name):
    if name == "陈老师":
        name = name + ", what do you want to teach in this class?"
    else:
        name = name + ", how are you?"
    print(name)
    return name

printsayhello("张三")
printsayhello("李四")
printsayhello("陈老师")