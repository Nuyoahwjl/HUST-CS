#include <stdio.h>

int main() {
/**********Begin**********/
	char c;
	scanf("%c",&c);
	if((c>='a'&&c<='f')||(c>='A'&&c<='F'))
	{
		switch (c){
			case 'a':
			case 'A': printf("10"); break;
			case 'b':
			case 'B': printf("11"); break;
			case 'c':
			case 'C': printf("12"); break;
			case 'd':
			case 'D': printf("13"); break;
			case 'e':
			case 'E': printf("14"); break;
			case 'f':
			case 'F': printf("15"); break;
		}
	} else if(c>='0'&&c<='9'){
		printf("%c",c);
	} else{
	    printf("%d",c);
	}
/**********End**********/
	return 0;
}