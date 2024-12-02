#include <iostream>
using namespace std;
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
 
    int main() {
        A a(2, 5), b(6), c;     
/*下面每条语句执行完后，i的值是多少，对象a，b，c的内容分别是多少*/
        int i = b.y;
        cout << i << endl;
        cout << a.x << " " << a.y << endl;  
        cout << b.x << " " << b.y << endl;
        cout << c.x << " " << c.y << endl;      

        i = a++;
        cout << i << endl;
        cout << a.x << " " << a.y << endl;  
        cout << b.x << " " << b.y << endl;
        cout << c.x << " " << c.y << endl; 

        i = a + c; 
        cout << i << endl;
        cout << a.x << " " << a.y << endl;  
        cout << b.x << " " << b.y << endl;
        cout << c.x << " " << c.y << endl;

        i = (b += c);
        cout << i << endl;
        cout << a.x << " " << a.y << endl;  
        cout << b.x << " " << b.y << endl;
        cout << c.x << " " << c.y << endl;
    } 