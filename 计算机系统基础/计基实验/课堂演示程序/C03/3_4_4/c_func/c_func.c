#include <stdio.h>
int fadd(int x, int y)
{
	int u,v,w;
	u = x + 10;
	v = y + 25;
	w = u + v;
	return w;
}

int main()
{
   int  a = 100;    // 0x 64
   int  b = 200;    // 0x C8
   int  sum = 0;
   sum = fadd(a, b);
   printf("%d\n", sum);
   return 0;
}
