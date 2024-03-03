/***在下面Begin至End间，按原型 void diceGame(int randSeed) 定义函数***/
/********** Begin **********/
void diceGame(int randSeed)
{
    int a,b,sum,k=1,summ;
    scanf("%d",&randSeed);
    srand(randSeed);
    a=rand();
    a=a%6+1;
    b=rand();
    b=b%6+1;
    sum=a+b;
    if(sum==7||sum==11)
    {
      printf("Round 1:  Score:%d  Success!\n",sum);
    }
    else 
    { 
        if(sum==2||sum==3||sum==12)
           {
             printf("Round 1:  Score:%d  Failed!\n",sum);goto out;
           }
           else
           {
             printf("Round 1:  Score:%d  Continue!\n",sum);
             printf("Next rounds: Score %d:Success, Score 7:Failed, others:Continue\n",sum);
           }
    
        for(k=2;k<=9;k++)
     {
        a=rand();
        a=a%6+1;
        b=rand();
        b=b%6+1;
        summ=a+b;
        if(summ==sum)
        {
            printf("Round %d:  Score:%d  Success!\n",k,summ);goto out;
        } 
        else 
        {
            if(summ==7)
            {
                printf("Round %d:  Score:%d  Failed!\n",k,summ);goto out;
            }
            else
            {
                printf("Round %d:  Score:%d  Continue!\n",k,summ);
            }
        }
             
     }
    }
    out:
    ;

    
}


/********** End **********/