#include "stdio.h"
#include "stdlib.h"
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


status ListLength(SqList L)
// 如果线性表L存在，返回线性表L的长度，否则返回INFEASIBLE。
{
    if(L.elem==NULL)
        return INFEASIBLE;
    else
        return L.length;
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
