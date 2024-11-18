#include <iostream>

using namespace std;
struct A { A() { cout << "A"; } };
struct B : A {
    A a;
    B() :A() { cout << "B"; }
};
struct C :virtual A {
    C() :A() { cout << "C"; }
};
struct D : virtual A, C {
    C c;
    D() :A(), C() { cout << "D"; }
};
struct E : virtual B, virtual C {
    E() :B(), C() { cout << "E"; }
};
struct F : virtual B, D, virtual E {
    D d;
    E e;
    F() :B(), D(), E() { cout << "F"; }
};
int main() {
    A a; cout << "\n";      //输出：
    B b; cout << "\n";      //输出：
    C c; cout << "\n";      //输出：
    D d; cout << "\n";      //输出：
    E e; cout << "\n";      //输出：
    F f; cout << "\n";      //输出：
}