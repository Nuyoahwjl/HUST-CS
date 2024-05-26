/********** Begin **********/
#include <stdio.h>
int main()
{
	int n;
	scanf("%d",&n);
	int x=1,y=1,z;
	int i;
	printf("%10d%10d",x,y);
	for(i=3;i<=n;i++){
		z=x+y;
		printf("%10d",z);
		x=y;
		y=z;
		if(i%8==0){
			printf("\n");
		}	
	}
	return 0;
 } 






/**********  End **********/