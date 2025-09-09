#include <stdio.h>

int main()
{
    int a = 0x80000000;
    int b = a/-1;
    printf("%d\n", b);
}