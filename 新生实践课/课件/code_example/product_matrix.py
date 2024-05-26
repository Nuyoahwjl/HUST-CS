# prodouct_matrix.py
import numpy as np # 导入numpy模块(包)

a=[[2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9]]
b = [5, 6]
a = np.array(a) # list转array
b = np.array(b) # list转array
output1 = np.dot(b[None, :], a) # b[None, :] # 增加维度
print(output1)
output1 = output1[-1, :] # 试试把-1改成0行不行? 为什么?
print(output1)