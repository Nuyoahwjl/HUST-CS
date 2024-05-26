# Class_Bear.py
class Bear:
    other_name = '老虎' # 属性
    def __init__(self, name = other_name): # 类的构造
        self.name = name # name: 类的属性
    def kill(self): # 类的方法
        print("%s, 是受保护动物, 不能杀..." % self.name)

animal = '狗熊'
bear = Bear() # 将类Bear实例化为对象bear
# bear.name = animal # 对象的属性
bear.kill() # 对象的方法
