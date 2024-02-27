//N
#include <stdio.h>
#include <math.h>
int main()
{
	int n;
	scanf("%d",& n);
	long long c[n+1];
	for(int i=1;i<n+1;i++) 
		scanf("%lld",&c[i]);
	int m,p1;
	long long s1,s2;
	scanf("%d %d %lld %lld",&m,&p1,&s1,&s2);
	long long bngg=0,hmgg=0;
	c[p1]+=s1;
	for(int i=1;i<=m;i++)
	{
		bngg+=c[i]*(m-i);
	}
	for(int i=n;i>=m;i--)
	{
		hmgg+=c[i]*(i-m);
	}
	long long gap=(long long)fabs(hmgg+(1-m)*s2-bngg);
	int flag=1;
	int p2;
	for(p2=2;p2<=n;p2++)
	{
		if((long long)fabs(hmgg+(p2-m)*s2-bngg)<gap)
		{
			gap=(long long)fabs(hmgg+(p2-m)*s2-bngg);
			flag=p2;
		}
	}
	printf("%d",flag);
	return 0;	
}
