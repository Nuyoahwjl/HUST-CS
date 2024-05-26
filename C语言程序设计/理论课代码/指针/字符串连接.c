/******************************************************************
 在下面编写带参 main0 函数，实现字符串的连接，字符串由命令行给出；
 连接之后的新串无冗余地存储到用`malloc`动态分配的空间，并将该空间首地址赋值给外部指针`p`。
 
 将本地调试通过的 main 改为 main0，可能需要按题目要求作适当修改，
 比如，这里的 main0 只完成连接操作，不需要输出数据。
 ******************************************************************/
 #include<string.h>
 #include<stdlib.h>
extern char *p;    // 外部指针的引用性声明，p指向连接后的串

int main0( int argc, char *argv[])
{
  /**********  Begin  **********/
//   int len;
//   for(int i=2;i<argc;i++){
//     len=strlen(argv[1]);
//     int j=0;
//     while(*(argv[i]+j)!='\0'){
//       *(argv[1]+len+j)=*(argv[i]+j);
//       j++;
//     }
//     len=strlen(argv[1]);
// *(argv[1]+len)='\0';
//   }
//   len=strlen(argv[1]);
//   p=(char *)malloc(len+1); 
//  p=argv[1];

int l=0;
for (int i=1;i<argc;i++){
  l+=strlen(argv[i]);
}
p=(char *)malloc(l+1);
int s=0;
for(int i=1;i<argc;i++){
  int len=strlen(argv[i]);
  for(int j=0;j<len;j++){
    *(p+s+j)=*(argv[i]+j);
  }
  s+=len;
}
*(p+s+1)='\0';
  /**********  End  **********/
}