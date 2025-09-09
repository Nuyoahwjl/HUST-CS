#include <stdio.h>

int main()
{
    int a = 0x80000000;
    int b = -1;
    int c = a/b;
    printf("%d\n", c);
}