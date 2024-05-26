
a = eval('3 * 7') # obj可以是字符串对象或者已经由compile编译过的代码对象
print('a=', a)
b = exec('3 * 7')
print('b=', b)
exec("print(3 * 7)")