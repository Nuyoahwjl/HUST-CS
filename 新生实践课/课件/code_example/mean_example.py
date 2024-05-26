# mean_example.py
import numpy as np

a=[[1, 2, 3, 4], [2, 3, 4, 5]]
a=np.array(a)
print(a)
print("mean for each row:", np. mean(a, axis=1)) # mean for each row
print("mean for each column:", np. mean(a, axis=0)) # mean for each column
print("mean for matrix:", np. mean(a[:])) # mean for matrix