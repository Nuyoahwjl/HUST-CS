#include <iostream>
using namespace std;

struct A
{
    A() { cout << "A"; }
};
struct B
{
    const A a;
    B() { cout << "B"; }
};
struct C : virtual A
{
    C() { cout << "C"; }
};
struct D
{
    D() { cout << "D"; }
};
struct E : A
{
    E() { cout << "E"; }
};
struct F : B, virtual C
{
    F() { cout << "F"; }
};
struct G : B
{
    G() : B() { cout << "G"; }
};
struct H : virtual C, virtual D
{
    H() { cout << "H"; }
};
struct I : E, F, virtual G, H
{
    E e;
    F f;
    I() { cout << "I"; }
};

int main()
{
    I i;
    return 0;
}