#include<stdio.h>
void connect(char *filename,char *filename1,char *filename2,char *filename3)
{
   FILE *fp,*taget;
   fp=fopen(filename,"w");
   char *s[3]={filename1,filename2,filename3};
   char ch[128];
   for(int i=0;i<3;i++){
   	taget=fopen(s[i],"r");
   	while(fgets(ch,128,taget)){
   		fputs(ch,fp);
	   }
	fclose(taget);
   }
   fclose(fp);
}