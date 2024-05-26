# nest.py
def function_1():
     print('正在调用function_1()...')
     def function_2():
        print('正在调用function_2()...')
     function_2()
function_1()
# function_2() #报错: 内嵌函数不能直接调用