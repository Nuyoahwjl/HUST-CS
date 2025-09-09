#include <stdio.h>

int main()
{
	int flag;
	int x = 3;
	int y = -1;
	if (x > 0 || y > 0)
		flag = 1;
	else 
		flag = 0;
	
	printf("flag = %d \n", flag);

	return 0;
}
