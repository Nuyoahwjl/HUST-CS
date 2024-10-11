#include<stdio.h>
#include<stdlib.h>
#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASIBLE -1
#define OVERFLOW -2
typedef int status;
typedef int ElemType; //数据元素类型定义
#define LIST_INIT_SIZE 100
#define LISTINCREMENT  10
typedef int ElemType;
typedef struct{  //顺序表（顺序结构）的定义
    ElemType * elem;
    int length;
    int listsize;
}SqList;
typedef struct{  //线性表的集合类型定义
     struct { char name[30];
               SqList L;    
     } elem[10];
     int length;
}LISTS;
LISTS Lists;     //线性表集合的定义Lists


status InitList(SqList &L)
// 线性表L不存在，构造一个空的线性表，返回OK，否则返回INFEASIBLE。
{
    if(L.elem==NULL)
    {
        L.elem=(int *)malloc(sizeof(int)*LIST_INIT_SIZE);
        L.listsize=LIST_INIT_SIZE;
        L.length=0;
        return OK;
    }
    else return INFEASIBLE;
}
{
    #include "def.h"
    #include "stu.h"
    int main()
    {
    SqList L;
    int i,j;
    scanf("%d",&i);
    if (!i)
        { 
        L.elem=NULL;
        j=InitList(L);
        if (L.elem==NULL)
            printf("可能没有正确分配元素空间");
        if (L.length)
            printf("未正确设置元素个数初始值");
        if (L.listsize!=LIST_INIT_SIZE)
            printf("未正确设置元素空间容量");
        if (j==OK) {
                printf("OK");
                L.elem[0]=1;
                L.elem[L.listsize-1]=2;
                }
        }
    else
        {
        L.elem=(ElemType *) malloc(sizeof(ElemType));
        j=InitList(L);
        if (j==INFEASIBLE) printf("INFEASIBLE");
        else printf("可能会对已经存在的线性表初始化");
        free(L.elem);
        }
    return 1;
    }
}

status DestroyList(SqList &L)
// 如果线性表L存在，销毁线性表L，释放数据元素的空间，返回OK，否则返回INFEASIBLE。
{
    if(L.elem!=NULL)
    {
        L.listsize=0;
        L.length=0;
        free(L.elem);
        L.elem=NULL;
        return OK;
    }
    else return INFEASIBLE;
}
{
    #include "def.h"
    #include "string.h"
    #include "ctype.h"
    #include "stu.h"


    status lex(FILE *fp,char word[])   //仅查找字母型标识符
    {
        char c,c1,i;
        while ((c=fgetc(fp))!=EOF && !feof(fp))
        {
            if (isspace(c)) continue;
            if (c=='/')
            {
                c==fgetc(fp);
                if (c==EOF) return ERROR;
                if (c=='/')
                {
                    while ((c=fgetc(fp))!='\n') 
                        if (c==EOF) return ERROR;
                    continue;
                }
                if (c=='*')
                {
                    c=fgetc(fp);
                    if (c==EOF) return ERROR;
                    do
                    {
                        c1=c;
                        c=fgetc(fp);
                        if (c==EOF) return ERROR;
                    } while (c1!='*' || c!='/');
                    continue;
                }
            }
            if (!isalpha(c)) continue;
            i=0;
            do {
                    word[i++]=c;
                } while (isalpha(c=fgetc(fp)));
            if (isspace(c)|| !(c>='0' && c<='9') || c==EOF)
            {
                word[i]='\0';
                return OK;
            }
        }
        return ERROR;
    }
    status match(char fileName[],char keyword[])
    {
        FILE *fp;
        char word[50];
        fp=fopen(fileName,"r");
        if (!fp) {printf("文件打开失败"); return FALSE;}
        while (lex(fp,word)==OK){
            if (strcmp(keyword,word))
                continue;
            fclose(fp);
            return TRUE;
        }
        fclose(fp);
        return FALSE;
    }

    int main()
    {
    SqList L;
    int f,i,j,e;
    scanf("%d",&f);
    if (f==0)
        {
            L.elem=NULL;
            j=DestroyList(L);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("不能对不存在的线性表进行销毁操作！");
        }
    else
        { 
            L.elem=(ElemType *) malloc(sizeof(ElemType)*10);
            L.length=0;
            L.listsize= 10;
        
            j=DestroyList(L);
            if (j==OK) 
                if (match("src/step02/stu.h","free")==FALSE || L.elem)
                    printf("未正确释放数据元素空间");
                else printf("OK");
            else printf("ERROR");
        }
    return 1;
    }
}

status ClearList(SqList &L)
// 如果线性表L存在，销毁线性表L，释放数据元素的空间，返回OK，否则返回INFEASIBLE。
{
    if(L.elem==NULL)
        return INFEASIBLE;
    else
    {
        L.length=0;
        return OK;
    }
}
{
    #include "def.h"
    #include "stu.h"
    int main() {
        SqList L;
        int i,j;
        scanf("%d",&i);
        if (!i) { 
            L.elem=NULL;
            j=ClearList(L);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表清空");
        }
        else {
            L.elem=(ElemType *) malloc(sizeof(ElemType));
            L.length = 1;
            j=ClearList(L);
            if (L.length) printf("未正确清空");
            if (!L.elem)  printf("错误释放元素空间");
            if (j==OK) printf("OK");
            free(L.elem);
        }
        return 1;
    }
}

status ListEmpty(SqList L)
// 如果线性表L存在，判断线性表L是否为空，空就返回TRUE，否则返回FALSE；如果线性表L不存在，返回INFEASIBLE。
{
    if(L.elem==NULL)
        return INFEASIBLE;
    else
    {
        if(L.length==0) return TRUE;
        else return FALSE;
    }
}
{
    #include "def.h"
    #include "stu.h"
    int main() {
        SqList L;
        int i,j;
        scanf("%d",&i);
        if (i==2) { 
            L.elem=(ElemType *) malloc(sizeof(ElemType));
            L.length=0;
            j=ListEmpty(L);
            if (j==OK) printf("TRUE");
            else printf("未正确判空");
            free(L.elem);
        }
        else if(i==1) {
            L.elem=(ElemType *) malloc(sizeof(ElemType));
            L.length=1;
            j=ListEmpty(L);
            if (j==ERROR) printf("FALSE");
            else printf("未正确判空");
            free(L.elem);
        }
        else {
            L.elem=NULL;
            L.length=0;
            j=ListEmpty(L);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表判空");
        }
        return 1;
    }
}

status ListLength(SqList L)
// 如果线性表L存在，返回线性表L的长度，否则返回INFEASIBLE。
{
    if(L.elem==NULL)
        return INFEASIBLE;
    else
        return L.length;
}
{
    #include "def.h"
    #include "stu.h"
    int main() {
        SqList L;
        int f,i,j,c=0;
        scanf("%d",&f);
        if (!f) {
            L.elem=NULL;
            L.length=10;
            j=ListLength(L);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表求表长");
        }
        else {
            L.elem=(ElemType *) malloc(sizeof(ElemType));
            scanf("%d",&i);
            while (i) {
            ++c;
            scanf("%d",&i);
            }
            L.length=c;
            j=ListLength(L);
            printf("%d", j);
            free(L.elem);
        }
        return 1;
    }
}

status GetElem(SqList L,int i,ElemType &e)
// 如果线性表L存在，获取线性表L的第i个元素，保存在e中，返回OK；如果i不合法，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
    if(L.elem==NULL)
        return INFEASIBLE;
    else if(i<=0||i>L.length)
        return ERROR;
        else{
            e=L.elem[i-1];
            return OK;
        }
}
{
    #include "def.h"
    #include "stu.h"
    int main() {
        SqList L;
        int f,i,j,e;
        scanf("%d",&f);
        if (!f) {
            L.elem=NULL;
            j=GetElem(L,2,e); 
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表获取元素");
            
        }
        else {
            L.elem=(ElemType *) malloc(sizeof(ElemType)*10);
            L.length=0;
            L.listsize= 100;
            scanf("%d",&i);
            while (i) {
                L.elem[L.length++]=i;
                scanf("%d",&i);
            }
            scanf("%d",&i);
            j=GetElem(L,i,e);
            if(j==OK) printf("OK\n%d",e);
            if(j==ERROR) printf("ERROR");
            free(L.elem);
        }
        return 1;
    }
}

int LocateElem(SqList L,ElemType e)
// 如果线性表L存在，查找元素e在线性表L中的位置序号并返回该序号；如果e不存在，返回0；当线性表L不存在时，返回INFEASIBLE（即-1）。
{
    if(L.elem==NULL)
        return INFEASIBLE;
    else
    {
        int i=0;
        for(i;i<L.length;i++)
        {
            if(L.elem[i]==e)
            return i+1;
        }
        if(i>=L.length) return ERROR;
    }
}
{
    #include "def.h"
    #include "stu.h"
    int main() {
        SqList L;
        int f,i,j,e;
        scanf("%d",&f);
        if (!f) {
            L.elem=NULL;
            L.length=3;
            j=LocateElem(L,e); 
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表查找元素");
            
        }
        else {
            L.elem=(ElemType *) malloc(sizeof(ElemType)*10);
            L.length=0;
            L.listsize= 10;
            scanf("%d",&i);
            while (i) {
                L.elem[L.length++]=i;
                scanf("%d",&i);
            }
            scanf("%d",&e);
            j=LocateElem(L,e);
            if(j==ERROR) printf("ERROR");
            else printf("%d",j);
            free(L.elem);
        }
        return 1;
    }
}

status PriorElem(SqList L,ElemType e,ElemType &pre)
// 如果线性表L存在，获取线性表L中元素e的前驱，保存在pre中，返回OK；如果没有前驱，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
    if(L.elem==NULL)
        return INFEASIBLE;
    else
    {
        int i=0;
        for(i;i<L.length;i++)
        {
            if(L.elem[i]==e)
            {
                if(i==0) return ERROR;
                else{
                    pre=L.elem[i-1];
                    return OK;
                }
            }
        }
        if(i>=L.length) return ERROR;
    }
}
{
    #include "def.h"
    #include "stu.h"
    int main() {
        SqList L;
        int f,i,j,e,pre;
        scanf("%d",&f);
        if (!f) {
            L.elem=NULL;
            L.length=3;
            j=PriorElem(L,e,pre); 
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表查找前驱元素");
        }
        else {
            L.elem=(ElemType *) malloc(sizeof(ElemType)*10);
            L.length=0;
            L.listsize= 10;
            scanf("%d",&i);
            while (i) {
                L.elem[L.length++]=i;
                scanf("%d",&i);
            }
            scanf("%d",&e);
            j=PriorElem(L,e,pre);
            if(j==ERROR) printf("ERROR");
            if(j==OK) printf("OK\n%d",pre);
            free(L.elem);
        }
        return 1;
    }
}

status NextElem(SqList L,ElemType e,ElemType &next)
// 如果线性表L存在，获取线性表L元素e的后继，保存在next中，返回OK；如果没有后继，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
    if(L.elem==NULL)
        return INFEASIBLE;
    else
    {
        int i=0;
        for(i;i<L.length;i++)
        {
            if(L.elem[i]==e)
            {
                if(i==L.length-1) return ERROR;
                else{
                    next=L.elem[i+1];
                    return OK;
                }
            }
        }
        if(i>=L.length) return ERROR;
    }
}
{
    #include "def.h"
    #include "stu.h"
    int main() {
        SqList L;
        int f,i,j,e,next;
        scanf("%d",&f);
        if (!f) {
            L.elem=NULL;
            L.length=3;
            j=NextElem(L,e,next); 
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表查找后继元素");
        }
        else {
            L.elem=(ElemType *) malloc(sizeof(ElemType)*10);
            L.length=0;
            L.listsize= 10;
            scanf("%d",&i);
            while (i) {
                L.elem[L.length++]=i;
                scanf("%d",&i);
            }
            scanf("%d",&e);
            j=NextElem(L,e,next);
            if(j==ERROR) printf("ERROR");
            if(j==OK) printf("OK\n%d",next);
            free(L.elem);
        }
        return 1;
    }
}

status ListInsert(SqList &L,int i,ElemType e)
// 如果线性表L存在，将元素e插入到线性表L的第i个元素之前，返回OK；当插入位置不正确时，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
    if(L.elem==NULL) return INFEASIBLE;
    if(i<=0||i>L.length+1) return ERROR;
    else{
        if(L.length>=L.listsize)
        {
            ElemType *newbase;
            newbase=(ElemType *)realloc(L.elem,sizeof(ElemType)*L.listsize+LISTINCREMENT);
            if(newbase)
            {
                L.elem=newbase;
                L.listsize+=LISTINCREMENT;
            }
        }
    }
    for(int j=L.length-1;j>=i-1;j--)
        L.elem[j+1]=L.elem[j];
    L.elem[i-1]=e;
    L.length++;
    return OK;
}
{
    #include "def.h"
    #include "stu.h"
    int main()
    {
    SqList L;
    int f,i,j,e;
    scanf("%d",&f);
    if (!f)
        {
            L.elem=NULL;
            j=ListInsert(L,1,1);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("不能对不存在的线性表进行插入操作！");
        }

    else
        {
        L.elem=(ElemType *) malloc(sizeof(ElemType)*10);
        L.length=0;
        L.listsize= 10;
        scanf("%d",&i);
        while (i)
        {
            L.elem[L.length++]=i;
            scanf("%d",&i);
        }
        scanf("%d%d",&i,&e);
        j=ListInsert(L,i,e);
        printf("%s\n", j==OK? "OK" : j==ERROR? "ERROR" : "OVERFLOW");
        for(i=0;i<L.length;i++)
            printf("%d ",L.elem[i]);
        }
    return 1;
    }
}

status ListDelete(SqList &L,int i,ElemType &e)
// 如果线性表L存在，删除线性表L的第i个元素，并保存在e中，返回OK；当删除位置不正确时，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
    if(L.elem==NULL) return INFEASIBLE;
    if(i<=0||i>L.length) return ERROR;
    e=L.elem[i-1];
    for(int j=i-1;j<L.length-1;j++)
        L.elem[j]=L.elem[j+1];
    L.length--;
    return OK;
}
{
    #include "def.h"
    #include "stu.h"
    int main() {
        SqList L;
        int f,i,j,e;
        scanf("%d",&f);
        if (!f) {
            L.elem=NULL;
            j=ListDelete(L,1,e); 
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("不能对不存在的线性表进行删除操作！");
        }
        else {
            L.elem=(ElemType *) malloc(sizeof(ElemType)*10);
            L.length=0;
            L.listsize= 10;
            scanf("%d",&i);
            while (i) {
                L.elem[L.length++]=i;
                scanf("%d",&i);
            }
            scanf("%d",&i);
            j=ListDelete(L,i,e);
            if(j==ERROR) printf("ERROR\n");
            if(j==OK) printf("OK\n%d\n",e);
            for(i=0;i<L.length;i++)
                printf("%d ",L.elem[i]);
            free(L.elem);
        }
        return 1;
    }
}

status ListTraverse(SqList L)
// 如果线性表L存在，依次显示线性表中的元素，每个元素间空一格，返回OK；如果线性表L不存在，返回INFEASIBLE。
{
    if(L.elem==NULL) return INFEASIBLE;
    if(L.length)
    {
        printf("%d",L.elem[0]);
        for(int i=1;i<L.length;i++)
            printf(" %d",L.elem[i]);
    }
    return OK;  
}
{
    #include "def.h"
    #include "stu.h"
    int main() {
        SqList L;
        int f,i,j;
        scanf("%d",&f);
        if (!f) {
            L.elem=NULL;
            j=ListTraverse(L); 
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表进行遍历操作！");
        }
        else {
            L.elem=(ElemType *) malloc(sizeof(ElemType)*10);
            L.length=0;
            L.listsize= 10;
            scanf("%d",&i);
            while (i) {
                L.elem[L.length++]=i;
                scanf("%d",&i);
            }
            j=ListTraverse(L);
            if(j==OK && !L.length) printf("空线性表\n");
            free(L.elem);
        }
        return 1;
    }
}

status  SaveList(SqList L,char FileName[])
// 如果线性表L存在，将线性表L的的元素写到FileName文件中，返回OK，否则返回INFEASIBLE。
{
    if(L.elem!=NULL){
        int i=0;
        FILE* fp=fopen(FileName,"w");
        if(fp==NULL){
            return ERROR;
        }
        else for(i=0;i<L.length;i++){
            fprintf(fp,"%d ",L.elem[i]);
        }
        fclose(fp);
        return OK;
    }
    else return INFEASIBLE;
}
status  LoadList(SqList &L,char FileName[])
// 如果线性表L不存在，将FileName文件中的数据读入到线性表L中，返回OK，否则返回INFEASIBLE。
{
    if(L.elem==NULL){
        int i=0,j,s[100];
        FILE*fp=fopen(FileName,"r");
        if(fp==NULL){
            return ERROR;
        }
        else L.length=0;
        while(fscanf(fp,"%d",&j)!=EOF){
            s[L.length++]=j;
        }
        L.elem=(ElemType*)malloc(L.listsize*sizeof(ElemType));
        for(i=0;i<L.length;i++){
            L.elem[i]=s[i];
        }
        fclose(fp);
        return OK;
    }
    else return INFEASIBLE;
}
{
    #include "def.h"
    #include "string.h"
    #include "stu.h"
    int main()
    {
    SqList L;
    FILE *fp;
    //char FileName[30];
    int f,i=0,j,e;
    //strcpy(FileName,"src/step13/list.dat");
    scanf("%d",&f);
    if (!f)
    {
            L.elem=NULL;
            j=SaveList(L,"src/step13/list.dat");
            if (j!=INFEASIBLE) printf("不能对不存在的线性表进行写文件操作！");
            else 
        {
                L.elem=(ElemType *) malloc(sizeof(ElemType));
                j=LoadList(L,"src/step13/list.dat");
                if (j!=INFEASIBLE) printf("不能对已存在的线性表进行读文件操作！");
                else printf("INFEASIBLE"); 
                free(L.elem);
        }
    }
    else
        {
            L.elem=(ElemType *) malloc(sizeof(ElemType)*LIST_INIT_SIZE);
            L.length=0;
            L.listsize= LIST_INIT_SIZE;
            scanf("%d",&e);
            while (e)
            {
                L.elem[i++]=e;
                scanf("%d",&e);
            }
            L.length=i;
            j=SaveList(L,"src/step13/list.dat");
            free(L.elem); 
            L.elem=NULL;
            j=LoadList(L,"src/step13/list.dat");
            printf("%d\n",L.length);
            for(i=0;i<L.length;i++) 
                printf("%d ",L.elem[i]);
        }
    return 1;
    }
}

status AddList(LISTS &Lists,char ListName[])
// 只需要在Lists中增加一个名称为ListName的空线性表，线性表数据又后台测试程序插入。
{
    Lists.length++;
    for(int i=0;i<30;i++)
        Lists.elem[Lists.length-1].name[i]=ListName[i];
    SqList LL;
    LL.elem=NULL;
    InitList(LL);
    Lists.elem[Lists.length-1].L=LL;
}
{  
    #include "def.h"
    #include "../step01/stu.h"
    #include "../step10/stu.h"
    #include "../step12/stu.h"
    #include "stu.h"
    int main() {
        LISTS Lists;
    int n,e;
    char name[30];
    Lists.length=0;
        scanf("%d", &n);
        while(n--)
    {
            scanf("%s",name);
            AddList(Lists,name);
        scanf("%d",&e);
        while (e)
        {
                ListInsert(Lists.elem[Lists.length-1].L,Lists.elem[Lists.length-1].L.length+1,e);
                scanf("%d",&e);
        }
    }
    for(n=0;n<Lists.length;n++)
    {
            printf("%s ",Lists.elem[n].name);
            ListTraverse(Lists.elem[n].L);
            putchar('\n');
    }
    return 1;
    }
}

status RemoveList(LISTS &Lists,char ListName[])
// Lists中删除一个名称为ListName的线性表
{
    for(int i=0;i<Lists.length;i++)
    {
        if(strcmp(Lists.elem[i].name,ListName)==0)
        {
            for(int j=i;j<Lists.length;j++)
            {
                Lists.elem[i]=Lists.elem[i+1];
            }
            Lists.length--;
            return OK;
        }
    }
    return ERROR;
}
{
    #include "def.h"
    #include "../step01/stu.h"
    #include "../step02/stu.h"
    #include "../step10/stu.h"
    #include "../step12/stu.h"
    #include "../step14/stu.h"
    #include "stu.h"
    int main() {
        LISTS Lists;
    int n,e;
    char name[30];
    Lists.length=0;
        scanf("%d", &n);
        while(n--)
    {
            scanf("%s",name);
            AddList(Lists,name);
        scanf("%d",&e);
        while (e)
        {
                ListInsert(Lists.elem[Lists.length-1].L,Lists.elem[Lists.length-1].L.length+1,e);
                scanf("%d",&e);
        }
    }
    scanf("%s",name);
    if (RemoveList(Lists,name)==OK)
        for(n=0;n<Lists.length;n++)
            {
                printf("%s ",Lists.elem[n].name);
                ListTraverse(Lists.elem[n].L);
                putchar('\n');
            }
    else printf("删除失败");
    return 1;
    }
}

int LocateList(LISTS Lists,char ListName[])
// 在Lists中查找一个名称为ListName的线性表，成功返回逻辑序号，否则返回0
{
    for(int i=0;i<Lists.length;i++)
    {
        if(strcmp(Lists.elem[i].name,ListName)==0)
        {
            return i+1;
        }
    }
    return 0;
}
{
    #include "def.h"
    #include "../step01/stu.h"
    #include "../step10/stu.h"
    #include "../step12/stu.h"
    #include "../step14/stu.h"
    #include "stu.h"
    int main() {
    LISTS Lists;
    int n,e;
    char name[30];
    Lists.length=0;
        scanf("%d", &n);
        while(n--)
    {
            scanf("%s",name);
            AddList(Lists,name);
        scanf("%d",&e);
        while (e)
        {
                ListInsert(Lists.elem[Lists.length-1].L,Lists.elem[Lists.length-1].L.length+1,e);
                scanf("%d",&e);
        }
    }
    scanf("%s",name);
    if (n=LocateList(Lists,name))
            {
                printf("%s ",Lists.elem[n-1].name);
                ListTraverse(Lists.elem[n-1].L);
            putchar('\n');
            }
    else printf("查找失败");
    return 1;
    }
}
