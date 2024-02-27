//I
#include <stdio.h>
#include <math.h>
int isprime(int n);
int main()
{
	int s,sum=0,num=0;
	scanf("%d",&s);
	for(int i=2;;i++){
		if(isprime(i)){
		sum+=i;
			if(sum<=s){
				num++;
				printf("%d\n",i);
			}
			else break;
		}
	}
	printf("%d",num);
	return 0; 
}

int isprime(int n)
{
	if(n==2) return 1;
	int flag=1;
	for(int i=2;i<=sqrt(n);i++){
		if(n%i==0){
			flag=0;
			break;
		}
	} 
	return flag;
}
