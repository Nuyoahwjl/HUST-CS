
/**
  按任务要求补全该函数
  用递归实现辗转相除法
 **/
int gcd(int x, int  y)
{
	/**********  Begin  **********/
   static int i = 0;
    if (i == 0)
    {
        printf("%d %d\n", x, y);
    }
    if (y == 0)
    {
        return x;
    }
    else
    {
 
        int r = x % y;
        printf("%d %d\n", y, r);
        i++;
        return gcd(y, r);
    }
 /**********  End  **********/
}