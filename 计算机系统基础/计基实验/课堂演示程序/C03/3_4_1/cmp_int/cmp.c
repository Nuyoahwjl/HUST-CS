#include <stdio.h>

void main()
{
    int x1 = 0;
    unsigned y1 = 0U;
    int r1 = (0 == 0U);
    r1 = x1 == y1;

    int x2 = -1;
    int y2 = 0;
    int r2 = (-1 < 0);
    r2 = x2 < y2;

    int x3 = -1;
    unsigned y3 = 0U;
    int r3 = (-1 < 0U);
    r3 = x3 < y3;

    int x4 = 2147483647;
    int y4 = -2147483647-1;
    int r4 = (2147483647 > -2147483647-1);
    r4 = x4 > y4;

    unsigned x5 = 2147483647U;
    int y5 = -2147483647-1;    
    int r5 = (2147483647U > -2147483647-1);
    r5 = x5 > y5;

    int x6 = 2147483647;
    int y6 = (int)2147483648U;   
    int r6 = (2147483647 > (int)2147483648U);
    r6 = x6 > y6;

    int x7 = -1;
    int y7 = -2;
    int r7 = (-1 > -2);
    r7 = x7 > y7;

    unsigned x8 = (unsigned)-1;
    int y8 = -2;
    int r8 = ((unsigned)-1 > -2);
    r8 = x8 > y8;    
}
