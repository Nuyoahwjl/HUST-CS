//H
#include <stdio.h>
int main()
{
	int n,grade;
	int min,max,sum=0;
	scanf("%d",&n);
	scanf("%d",&grade);
		sum+=grade;
		max=grade;
		min=grade;
	for(int i=1;i<n;i++){
		scanf("%d",&grade);
		sum+=grade;
		if(grade>max) max=grade;
		if(grade<min) min=grade;
	}
	sum=sum-max-min;
	printf("%.2f",1.0*sum/(n-2));
	return 0;
}
