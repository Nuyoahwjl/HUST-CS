//AL
#include <stdio.h>
int main()
{
	int n;
	long long m;
	scanf("%d%lld",&n,&m);
	int a[n+1];
	for(int i=1;i<=n;i++)
		scanf("%d",&a[i]);
	long long c[n+1];
	c[0]=1;
	for(int i=1;i<=n;i++)
	{
		c[i]=c[i-1]*a[i];
	}
	long long sum=0;
	int b[n+1];
	for(int i=1;i<=n;i++)
	{
		b[i]=(m%c[i]-sum)/c[i-1];
		sum+=c[i-1]*b[i]; 
	}
	for(int i=1;i<=n;i++)
		printf("%d ",b[i]);
}
