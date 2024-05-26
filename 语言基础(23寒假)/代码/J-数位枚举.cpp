//J
#include <stdio.h>
int main()
{
	int n,x;
	int num=0;
	scanf("%d%d",&n,&x);
	for(int i=1;i<=n;i++){
		for(int j=i;j!=0;j/=10){
			if(j%10==x) num++;
		}
	}
	printf("%d",num);
	return 0;
}
