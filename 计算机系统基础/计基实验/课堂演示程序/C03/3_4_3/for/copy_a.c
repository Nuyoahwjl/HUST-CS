#include <stdio.h>
int main()
{
    char    buf1[20];
    char    buf2[20];
    int     i;
    scanf("%s", buf1);
    for (i = 0; i < 20; i++)
        buf2[i] = buf1[i];
	printf("%s\n", buf2);
	return 0;
}