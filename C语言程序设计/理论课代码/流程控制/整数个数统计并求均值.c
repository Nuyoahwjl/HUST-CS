/********** Begin **********/
#include <stdio.h>
int main()
{
    int i=1;
    int x;
    int numbers=0;
    int sum=0;
    float average;
    for(i=1;i<=10;i++){
        scanf("%d",&x);
        if(x>0) {
            numbers++;
            sum+=x;
        } else continue;
        
        }
             if(numbers==0){
                 printf("numbers=0, average=0.000000");
             }else {
                 printf("numbers=%d, average=%f",numbers,1.0*sum/numbers);
             }
             return 0;
}






/**********  End **********/