/********** Begin *********/
#include <stdio.h>
int main()
{
    double x,y;
    char c;
    scanf("%lf %lf %c",&x,&y,&c);
    switch(c){
        case '+':
            printf("%.1lf\n", x + y); break;
        case '-':
            printf("%.1f\n", x - y); break;
        case '*':
            printf("%.1f\n", x * y); break;
        case '/':   
            printf("%.1f\n", x / y); break;
        }
    return 0;
}






/**********  End **********/