//B
#include <stdio.h>
#include <math.h>
int main()
{
	float s,ss;
	scanf("%f",&s);
	int i;
	for(i=1;ss<s;i++){
		if(ss=100*(1-pow(0.98,i))>=s)
		break;
	}
	printf("%d",i);
	return 0;
}
