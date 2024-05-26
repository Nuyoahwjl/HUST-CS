/********** Begin **********/
#include <stdio.h>
int main()
{
	double e;
	double PI=0.0;
	scanf("%lf",&e);
	int a=1;
	int n=1;
	
	do{
		PI=PI+a*1.0/n;
		a=-a;
		n+=2;
	}while(n<=1.0/e+2);
	printf("%lf",4*PI);
	return 0;
	
	
}






/**********  End **********/