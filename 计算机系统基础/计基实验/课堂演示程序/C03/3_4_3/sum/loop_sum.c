#include <stdio.h>

int main()
{
    int n = 10;
    int i;
    int result = 0;
    
	for (i=1; i <= n; i++)  
	      result += i; 

    printf("%d\n", result);
    return 0;
}
