#include<stdio.h>
struct date{
	int year;
	int month;
	int day;
};
int main()
{
	
	int sum=0;
	struct date t;
	scanf("%d%d%d",&t.year,&t.month,&t.day );
	int a[12]={31,28,31,30,31,30,31,31,30,31,30,31};
    if(t.year%4==0&&t.year%100!=0||t.year%400==0) a[1]++;
    if(t.month<1||t.month>12||t.day>a[t.month-1]) printf("不存在这样的日期");
    else{
    	for(int i=0;i<t.month-1;i++) sum+=a[i];
    	sum+=t.day;
    	printf("%d",sum);
	}
	

	return 0; 
}