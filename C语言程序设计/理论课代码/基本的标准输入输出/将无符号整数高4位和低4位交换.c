#include <stdio.h>

int main(){
/**********Begin**********/
    unsigned short k,a,b,c;
	scanf("%hu",&k);
	a=k>>12;
	b=k<<12;
	c=k&0x0ff0;
	k=a^b^c;
	printf("%hu",k);
/**********End**********/
	return 0;
}