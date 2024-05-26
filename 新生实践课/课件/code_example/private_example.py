# private.py
def _private_1(name):
    return 'Hello, %s' % name

def _private_2(name):
    return 'Hi, %s' % name

def greeting(name): # 外部需要的函数为greeting(name)
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)
name = "陈加忠大侠"
print(greeting(name))