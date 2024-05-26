# return_a_function.py
# 定义求和函数, 返回的并不是求和结果, 而是计算求和的函数
def lazy_plus(*args):
    def plus():
        s = 0
        for n in args:
            s = s + n
        return s
    return plus

# 调用函数f时, 得到真正求和的结果
f = lazy_plus(1, 2, 3, 4, 5)
print(f()) # 改成print(f)试试?
