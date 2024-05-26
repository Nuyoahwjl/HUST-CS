//AD
#include <stdio.h>
int main()
{
	int n,m;
	scanf("%d %d",&n,&m);
	int a[n+1][m+1];
	for(int i=1;i<n+1;i++)
	{
		for(int j=1;j<m+1;j++)
		{
			scanf("%d",&a[i][j]);
		}
	}
	for(int i=1;i<n+1;i++)
	{
		for(int j=1;j<n+1;j++)
		{
			int flag=1;
			for(int k=1;k<m+1;k++)
			{
				if(a[i][k]>=a[j][k])
				{
					flag=0;
					break;
				}
			}
			if(flag==1) 
			{
			printf("%d\n",j);
			break;
			}
			if(flag==0&&j==n) printf("0\n");
		}
	}
}
