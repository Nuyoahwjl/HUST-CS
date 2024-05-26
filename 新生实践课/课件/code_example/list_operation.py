# list_operation.py

guests=['Zhang san', 'Li si', 'Wang wu', 'Zhao liu']
print("original input:", guests)
guests.insert(1, 'jiazhong chen') # 第二个位置插入
print("after insert:", guests)
guests.append('hu qi') # 最后位置添加
print("after append:", guests)
guests[2]='wang fang' # 把第三位置的改写
print("after modify:", guests)
del guests[2] # 按位置删掉
print("after delete:", guests)
name = guests.pop(2) # 按位置删除, 但返回被删除的值
print("after pop:", guests)
print("name pop:", name)
guests.remove('jiazhong chen') # 按值删除
print("after remove:", guests)