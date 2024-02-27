//D
#include <stdio.h>
int main()
{
	int x;
	scanf("%d",&x);
	int a,b;
	a=!(x%2);
	if(x>4&&x<=12) b=1;
	else b=0;
	if(a&&b) printf("1 ");
	else printf("0 ");
	if(a||b) printf("1 ");
	else printf("0 ");
	if(a&&!b||!a&&b) printf("1 ");
	else printf("0 ");
	if(!a&&!b) printf("1 ");
	else printf("0 ");
    return 0;
}
