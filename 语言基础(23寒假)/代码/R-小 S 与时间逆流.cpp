//R
#include <stdio.h>
#include <string.h>
char * reverse(char *s,int left,int right) 
{
    while (left<right) 
	{
        char temp=s[left];
        s[left]=s[right];
        s[right]=temp;
        left++;
        right--;
    }
    return s;
}

int main() 
{
    char record[101];
    scanf("%s",record);
    int n=strlen(record);
    int left=0,right=n-1;
    for (int i=0;i<n; i++) 
	{
        if (record[i]=='1') 
		{
            left=i;
            right=i+1;
            break;
        }
    }
    char min[101];
	strcpy(min,record);
    for(int i=right;i<n;i++)
    {
    	char s1[101];
    	strcpy(s1,record);
		char temp[101];
		strcpy(temp,reverse(s1,left,i));
    	if((strcmp(temp,min))<0)
    	{
    		strcpy(min,temp);
		}
	}
    printf("%s",min);
    return 0;
}
