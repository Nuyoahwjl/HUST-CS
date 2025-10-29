#include <stdio.h>

int main()
{
    unsigned int a = 1;
    unsigned short b = 1;
    char c = -1;
    int d;

    d = (a > c) ? 1 : 0;
    printf("%d\n", d);
    d = (b > c) ? 1 : 0;
    printf("%d\n", d);
}