#include <stdio.h>
#include <string.h>
int main()
{
	char s[128];
	FILE *fp;
	int flag=0;
	int cnt=0;
    scanf("%s",s);
	if(strcmp(s,"type_c")!=0){
		printf("鎸囦护閿欒  \n");
		return 0;
	}
    scanf("%s",s);
	if(strcmp(s,"/p")==0) 
        flag=1;
	fp=fopen("src/step1_1/test1.c", "r");
    while (fgets(s, 128, fp))
    {
        printf("%d  %s", ++cnt, s);
        if (flag && cnt == 10)
        {
            cnt = 0;
            while (scanf("%s", s) && strcmp(s, "q"))
                continue;
        }
    }
    fclose(fp);
	return 0; 
}