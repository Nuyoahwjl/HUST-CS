#include<stdio.h>
int main()
{
	int i=1;
	int sum=0;
	int n=0;
	int x;
for (i=1;i<=10;i++){
    scanf("%d",&x);
    if (x>0){
        sum+=x;
        n++;
    }
}
if (n>0){
printf("累加和:%d\n",sum);
printf("平均值:%.1lf\n",1.0*sum/n);
} else {
	printf("累加和:0\n");
    printf("平均值:0.0\n");
}
    return 0;
}