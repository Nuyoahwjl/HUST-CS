//AF
#include <stdio.h>
int main()
{
	int n,m;
	scanf("%d%d",&n,&m);
	int a[n+1][2];
	a[n][0]=a[n][1]=0;
	for(int i=0;i<n;i++)
	{
		scanf("%d%d",&a[i][0],&a[i][1]);
		a[n][0]+=a[i][0];
		a[n][1]+=a[i][1];
	}
	int b[m][2];
	for(int i=0;i<m;i++)
	{
		scanf("%d%d",&b[i][0],&b[i][1]);
	}
	for(int i=0;i<m;i++)
		printf("%d %d\n",b[i][0]+a[n][0],b[i][1]+a[n][1]);
	return 0;
}
