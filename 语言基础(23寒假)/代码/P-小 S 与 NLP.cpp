//P
#include <stdio.h>
#include <string.h>
int main()
{
	char ch;
	int n,m;
	scanf("%d %d",&n,&m);
	char a[n][30];
	int b[n];
	for(int i=0;i<n;i++)
	{
		scanf("%s %d",a[i],b+i);
	}
	getchar();
	char c[m][100000];
	for(int i=0;i<m;i++)
	{
		fgets(c[i],100010,stdin);
//		getchar();
//		int j=0;
//		while((ch=getchar())!='\n')
//		{
//			c[i][j]=ch;
//			j++;
//		}
	}
	for(int i=0;i<m;i++)
	{
		int l=strlen(c[i]);
		for(int j=0;j<l;j++)
		{
			if(c[i][j]=='{')
			{
				int k;
				for(k=j+1;c[i][k]!='}';k++)
				{
				}
				c[i][k]='\0';
				for(int p=0;p<n;p++)
				{
					if(!(strcmp(a[p],&c[i][j+1])))
					{
//						c[i][j]=b[p]+'0';
						printf("%d",b[p]);
//						for(int q=j+1;q<l+1;q++)
//						{
//							c[i][q]=c[i][q+strlen(a[p])+1];
//						}
					}
				}
				j=k;
			}
			else printf("%c",c[i][j]);
		}
//		printf("%s",c[i]);
	}
	return 0;
}
