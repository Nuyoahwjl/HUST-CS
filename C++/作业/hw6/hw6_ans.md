## 第1题
``` cpp
struct A {
    static int  *j;
    static int  A::*a;        
    static int  i[5];
    int x;
    static int  &k;
    static int  *n;
};
int  y = 0;
/*
*   请基于以上代码，在A类的类体外初始化5个静态数据成员
*   请不要额外定义任何其它变量
*/
```
``` cpp
int *A::j = &y;
int A::A::*a = &A::x;
int A::i[5] = {1, 2, 3, 4, 5};
int &A::k = A::x;
int *A::n = &A::i[0];
```

---

## 第2题
``` cpp
/*
*   请指出下面代码存在的错误，并说明原因  
*/
struct A {
    int i;
    static const int  j = 0;
    static const double  d = 0.0;  
    static void f(A a) {
        int x = a.i;
        int y = i + a.i;  
    }
};    
```
``` cpp
// static const double d = 0.0;
// static const类型的成员只能是整型或枚举类型才能在类中直接进行初始化。
// double类型的静态常量需要在类外进行定义和初始化。

// int y = i + a.i;
// i是A类的非静态成员变量，不能在静态方法f中直接使用，因为静态方法中没有this指针
```