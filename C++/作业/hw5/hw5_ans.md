## 第1题
``` cpp
/*阅读下面程序并回答问题*/
    struct A {
        int i;
        int j;
        operator int() { return i + j; }
        A(int x, int y):i(x),j(y){}
    } a = { 1,2 };

    struct B :A {
        int m;
        int n;
        operator A() { return A(i, j); }
        operator int() { return A(*this) + m + n; }
        B(int x, int y, int m, int n) :A(x, y), m(m), n(n){}
    } b = { a,a,3,4 };

    void test() {
/*下面语句执行以后，i的值是多少？请说明详细计算过程*/
        int i = a + b; 
        std::cout << "i = " << i << std::endl;
    }
```
``` cpp
i = 16
/*
A a = {1, 2}; → i = 1, j = 2
B b = {a, a, 3, 4}; → a被转换成int类型传入B的构造函数
                    → 调用A::operator int(), int(a) = 3 
                    → x = y = 3
                    → B类中 A::i=3, A::j=3, B::m=3, B::n=4
int i = a + b; → a被转换成int类型,int(a) = 3
               → b被转换成int类型,调用B::operator int()
               → A(*this)调用B::operator A()返回匿名对象A(3,3)
               → 匿名对象被转换为int类型,int(A(3,3)) = 6
               → int(b) = 6 + 3 + 4 =13
               → i = 16
*/
```

----------

## 第2题
``` cpp
/*阅读下面程序并回答问题*/
    struct A {
        int x;
        static int y;
    public:
        operator int() { return x + y; }
        A& operator+=(const A& a);
        A operator++(int);
        A(int x = 1, int y = 1) :x(x) { A::y = y; }
    };

    int A::y = 20;
    A& A::operator+=(const A& a) {
        x += a.x;
        y += a.y;
        return *this;
    }
    A A::operator++(int) {
        return A(x++, y++);
    }
 
    void test() {
        A a(2, 5), b(6), c;     
/*下面每条语句执行完后，i的值是多少，对象a，b，c的内容分别是多少*/
        int i = b.y;        
        i = a++;            
        i = a + c;          
        i = (b += c);
    }      
```
``` cpp
int i = b.y;
// i=1;
// a.x=2; a.y=1;
// b.x=6; b.y=1;
// c.x=1; c.y=1;
i = a++;
// i=5;
// a.x=3; a.y=2;
// b.x=6; b.y=2;
// c.x=1; c.y=2;
i = a + c;
// i=8;
// a.x=3; a.y=2;
// b.x=6; b.y=2;
// c.x=1; c.y=2;
i = (b += c);
// i=11;
// a.x=3; a.y=4;
// b.x=7; b.y=4;
// c.x=1; c.y=4;
```