
def Fibonacci(n):
   """
   输入： 
      n：斐波那契数列的阶数
   输出：返回两个结果
      Fib[n]：第n个斐波那契数列的值
      count：迭代的次数
   """
   Fib = [0,1]
   count = 1
   for i in range(2,n+1):
      Fib.append(Fib[i-2]+Fib[i-1])
      count+=1

   return Fib[n], count