#include <stdio.h>
int main()
{
	int i = 1;
	int n = 100;
	int result = 0;
	while (i <= n)
	{
		result += i;
		i++;
	}

	printf("result = %d \n", result);
	return 0;
}