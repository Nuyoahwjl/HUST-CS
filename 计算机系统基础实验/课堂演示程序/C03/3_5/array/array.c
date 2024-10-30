#include <stdio.h>

int main()
{
    int     a[10];
    char    b[50] = "1234567";

    for (int i = 0; i < 10; i++)
        a[i] = i * 3;

    for (int j = 0; j < 15; j++)
        putchar(b[j]);    

    return 0;
}

