#include <stdio.h>

int main()
{
	int  x = 3;
	int  y = -1;
	int  z;
	int  i = 1;
	char c;
	c = getchar();
	switch (c) {
	case '+':
	case 'a':
		z = x + y;
		break;
	case '-':
	case 's':
		z = x - y;
		break;
	default:
		z = 0;
	}
	printf(" %d %c %d = %d \n", x, c, y, z);
	return 0;
}
