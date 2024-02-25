//O
#include <stdio.h> 
int main()
{
	int n,k;
	scanf("%d %d",&n,&k);
	int a[n];
	for(int i=0;i<n;i++)
		scanf("%d",a+i);
	int temp;
	for(int i=0;i<n-1;i++)
	{
		for(int j=0;j<n-i-1;j++)
		{
			if(a[j]>a[j+1])
			{
				temp=a[j];
				a[j]=a[j+1];
				a[j+1]=temp;
			}
		}
	} 
	int num=0;
	for(int i=0;i<n-1;i++)
	{
		if(a[i]==a[i+1])
		num++;
	}
	if(k>n-num||k<=0) printf("NO RESULT");
	else 
	{
		for(int i=0;i<n-1;i++)
		{
			if(a[i]==a[i+1])
			{
				for(int j=i+1;a[j]==a[i];j++)
					a[j]=0;
			}
		}
		int p=0;
		for(int i=0;i<n;i++)
		{
			if(a[i]) p++;
			if(p==k)
			{
				printf("%d",a[i]);
				break;
			}
		}
	}
	return 0;
 } 
