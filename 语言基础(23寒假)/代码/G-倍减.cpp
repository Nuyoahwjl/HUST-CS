//G
#include <stdio.h>
int main()
{
	int x;
	int num;
	scanf("%d",&x);
	for(num=0;x!=1;x/=2) num++;
	printf("%d",num+1);
	return 0;
}
