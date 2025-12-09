#include <stdio.h>

int main()
{
    char a[3][5];
    char *p00 = &(a[0][0]);
    char *p01 = &(a[0][1]);
    char *p02 = &(a[0][2]);
    char *p10 = &(a[1][0]);
    char *p11 = &(a[1][1]);
    char *p20 = &(a[2][0]);
    char *p24 = &(a[2][4]);
    
    return 0;
}