//L
#include <stdio.h>
#include <math.h>
int main()
{
	int n,q;
	scanf("%d%d",&n,&q);
	int a[n],b[q][2];
	int temp;
	for(int i=0;i<n;i++) scanf("%d",&a[i]);
	for(int i=0;i<q;i++) scanf("%d%d",&b[i][0],&b[i][1]);
	//½«ÌØÕ÷ÂëÅÅÐò
	for(int i=0;i<n-1;i++){
		for(int j=0;j<n-i-1;j++){
			if(a[j]>a[j+1]){
				temp=a[j];
				a[j]=a[j+1];
				a[j+1]=temp;
			}
		}
	} 
	for(int i=0;i<q;i++){
		int m=b[i][0];
		int k=b[i][1];
		int flag=0;
		for(int j=0;j<n;j++){
			if((a[j]%(int)(pow(10,m)))==k){
				flag=1;
				printf("%d\n",a[j]);
				break;
			}
		}
		if(flag==0) printf("-1\n");
	}
	return 0;
 } 
