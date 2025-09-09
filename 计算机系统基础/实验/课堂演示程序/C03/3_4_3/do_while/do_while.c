#include <stdio.h>
int main()
{
	int i = 1;
	int result = 0;
	do
	{
		result += i;
		i++;
	} 
	while (i <= 10);

	printf("result = %d \n", result);
	return 0;
}