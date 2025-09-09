#include <stdio.h>

int main()
{
	int a;
	int x;
	scanf("%d", &a);
	x = a > 0 ? a : a + 100;
	printf("x = %d \n", x);
	return 0;
}