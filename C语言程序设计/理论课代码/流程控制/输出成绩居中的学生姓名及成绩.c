/********** Begin **********/
#include <stdio.h>
int main()
{
	int a,b,c;
	scanf("%d %d %d",&a,&b,&c);
	if (a==b&&b==c){
		printf("A %d",a);
	}
	if (a==b&&a!=c){
		printf("A %d",a);
	}
	if (a==c&&a!=b){
		printf("A %d",a);
	}
	if (b==c&&b!=a){
		printf("B %d",b);
	}
	
    if (a>b&&b>c){
		printf("B %d",b);
	}
	if (a>c&&c>b){
		printf("C %d",c);
	}
	if (b>a&&a>c){
		printf("A %d",a);
	}
	if (b>c&&c>a){
		printf("C %d",c);
	}
	if (c>a&&a>b){
		printf("A %d",a);
	}
	if (c>b&&b>a){
		printf("B %d",b);
	}
	return 0;
}






/**********  End **********/