/********** Begin **********/
#include <stdio.h>
int main()
{
	int n;
	scanf("%d",&n);
	int y,a,b,c;
	int i=0;
	for(y=1;y<n;y++){
		for(a=1;a<100;a++){
			for(b=1;b<32;b++){
				for(c=1;c<100;c++){
					if(y==a*a&&y==b*b*10+c*c){
						printf("%d=%d*%d=%d*%d*10+%d*%d\n",y,a,a,b,b,c,c);
						i=1;
					}
				} if(i==1) {
					i=0;
					break;
				}
			} 
		}
	}
	return 0;
}





/**********  End **********/