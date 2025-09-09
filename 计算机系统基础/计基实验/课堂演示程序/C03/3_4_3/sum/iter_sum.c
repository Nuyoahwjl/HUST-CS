#include <stdio.h>

int iter_sum(int n)
{
	int result;	
	if  (n <= 0)  
	    result = 0;   
	else
	    result = n + iter_sum(n - 1); 
	return  result;
}

int main()
{
    int sum = iter_sum(10);
    printf("%d\n", sum);
	return 0;
}
