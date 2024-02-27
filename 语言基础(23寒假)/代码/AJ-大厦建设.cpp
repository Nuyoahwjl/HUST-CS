//AJ
#include <stdio.h>
typedef long long ll;
int main()
{
	ll N,a,b;
	scanf("%lld%lld%lld",&N,&a,&b);
	ll sum=0;
	for(ll i=0;i<N;i++)
	{
		ll x1,y1,x2,y2;
		scanf("%lld%lld%lld%lld",&x1,&y1,&x2,&y2);
		if(x1>=a||y1>=b||x2<=0||y2<=0)
		{
		}
		else
		{
		ll gapx1=x1>0?x1:0;
		ll gapy1=y1>0?y1:0;
		ll gapx2=x2>a?a:x2;
		ll gapy2=y2>b?b:y2;
		sum+=(gapx2-gapx1)*(gapy2-gapy1);
		}
	}
	printf("%lld",sum);
	return 0;
}
