#include <stdio.h>

void main()
{
    int    x = 0x40000001;
    float  f = 1.2;
    double d = 1.78;

    printf("%f, %d, %d\n", (float)x, x, x == (int)(float) x);
    printf("%d\n", x == (int)(double) x);
    printf("%d\n", f == (float)(double) f);
    printf("%d\n", d == (float) d);
    printf("%d\n", f == -(-f));
    printf("%d\n", 2/3 == 2/3.0);
    printf("%d\n", d < 0.0 == ((d*2) < 0.0));
    printf("%d\n", d > f == -f > -d);
    printf("%d\n", d * d >= 0.0);
    printf("%d\n", x * x >= 0);
    printf("%d\n", (d + f) - d == f);
}
