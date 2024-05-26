# tuple_example
tuple_in = ('kk', 768, 2.23, 'json', 70.2)
tintuple = (123, 'json')
print(tuple_in)
print(tuple_in[0])
print(tuple_in[1:3])
print(tuple_in[2:])
print(tintuple*2)
print(tuple_in+tintuple)
# tuple convert to list
a = list(tuple_in)
print("tuple convert to list:", a)
# list convert to tuple
a = tuple(a)
print("list convert to tuple:", a)