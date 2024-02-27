//AK
#include <stdio.h>
int findmax(int t[],int c[],int n)
{
	int max=t[0];
	int time=c[0];
	int flag=0;
	for(int i=1;i<n;i++)
	{
		if(t[i]>max) 
		{
			max=t[i];
			time=c[i];
			flag=i;
		}
		if(t[i]==max)
		{
			if(c[i]<time)
			{
				flag=i;
			}
		}
	}
	return flag;
}
int main()
{
	int n,k;
	long long m;
	scanf("%d%lld%d",&n,&m,&k);
	int t[n],c[n];
	for(int i=0;i<n;i++)
		scanf("%d%d",&t[i],&c[i]);
	while(m!=0)
	{
		int p=findmax(t,c,n);
		if(c[p]<=m&&t[p]>k)
		{
		t[p]-=1;
		m-=c[p];
		}
		else break;
	}
	int j=findmax(t,c,n);
	printf("%d",t[j]);
	return 0;
}
