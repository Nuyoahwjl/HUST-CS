#include <stdio.h>
int main()
{
	int n = 100;
	int result = 0;
	for (int i = 1; i <= n; i++) 
		result += i;
	printf("result = %d \n", result);
	return 0;
}