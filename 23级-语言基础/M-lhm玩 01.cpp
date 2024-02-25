//M
#include <stdio.h>
int main()
{
	int N;
	scanf("%d",&N);
	int n,flag=0,num=0,res=0;
	do{
		scanf("%d",&n);
		flag++;
		num+=n;
		for(int i=0;i<n;i++){
			if(flag%2){
				printf("0");
				res++;
			}
			else{
				printf("1");
				res++;
			}
			if(res%N==0) printf("\n");
		}
	}while(num!=N*N);
	return 0;
}
