#include "stdio.h"
#include "stdlib.h"
using namespace std;
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
typedef struct LNode{  //单链表（链式结构）结点的定义
      ElemType data;
      struct LNode *next;
}LNode,*LinkList;

// int main()
// {
//     LinkList L=NULL;
//     InitList(L);
//     if(L==NULL) printf("no");
//     else printf("yes");
//     return 0;
// }

status InitList(LinkList &L)
// 线性表L不存在，构造一个空的线性表，返回OK，否则返回INFEASIBLE。
{
    if(L==NUll)
    {
        L=(LinkList)malloc(sizeof(LNode));
        L->next=NULL;
        return OK;
    }
    else return INFEASIBLE;
}


status DestroyList(LinkList &L)
// 如果线性表L存在，销毁线性表L，释放数据元素的空间，返回OK，否则返回INFEASIBLE。
{
    if(L)
    {
        LinkList p=L;
        while(L)
        {
            L=L->next;
            free(p);
            p=L;
        }
        return OK;
    }
    else return INFEASIBLE;
}


status ClearList(LinkList &L)
// 如果线性表L存在，删除线性表L中的所有元素，返回OK，否则返回INFEASIBLE。
{
    if(L)
    {
        LinkList p=L->next;
	    L->next=NULL;
	    DestroyList(p);
        return OK;
    }
    else return INFEASIBLE;
}


status ListEmpty(LinkList L)
// 如果线性表L存在，判断线性表L是否为空，空就返回TRUE，否则返回FALSE；如果线性表L不存在，返回INFEASIBLE。
{
    if(L)
    {
        if(L->next) return FALSE;
        else return TRUE;
    }
    else return INFEASIBLE;
}


int ListLength(LinkList L)
// 如果线性表L存在，返回线性表L的长度，否则返回INFEASIBLE。
{
    if(L)
    {
        int i=0;
        while(L->next)
        {
            i++;
            L=L->next;
        }
        return i;
    }
    else return INFEASIBLE;
}


status GetElem(LinkList L,int i,ElemType &e)
// 如果线性表L存在，获取线性表L的第i个元素，保存在e中，返回OK；如果i不合法，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
    if(L)
    {
        int len=ListLength(L);
        if(i<=0||i>len) return ERROR;
        else 
        {
            while(i--) L=L->next;
            e=L->data;
            return OK;
        }
    }
    else return INFEASIBLE;
}


status LocateElem(LinkList L,ElemType e)
// 如果线性表L存在，查找元素e在线性表L中的位置序号；如果e不存在，返回ERROR；当线性表L不存在时，返回INFEASIBLE。
{
    if(L)
    {
        int i=0;
        L=L->next;
        while(L)
        {
            i++;
            if(L->data==e) return i;
            L=L->next;
        }
        return ERROR;
    }
    else return INFEASIBLE;
}


status PriorElem(LinkList L,ElemType e,ElemType &pre)
// 如果线性表L存在，获取线性表L中元素e的前驱，保存在pre中，返回OK；如果没有前驱，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
    if(L)
    {
        if(L->next==NULL) return ERROR;
        if(L->next->data==e) return ERROR;
        else
        {
            L=L->next;
            while(L->next)
            {
                if(L->next->data==e) 
                {
                    pre=L->data;
                    return OK;
                }
                L=L->next;
            }
            return ERROR;
        }
    }
    else return INFEASIBLE;
}


status NextElem(LinkList L,ElemType e,ElemType &next)
// 如果线性表L存在，获取线性表L元素e的后继，保存在next中，返回OK；如果没有后继，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
    if(L)
    {
        if(L->next==NULL) return ERROR;
        L=L->next;
        while(L)
        {
            if(L->data==e)
            {
                if(L->next==NULL) return ERROR;
                else{
                    next=L->next->data;
                    return OK;
                }
            }
            L=L->next;
        }
        return ERROR;
    }
    else return INFEASIBLE;
}


status ListInsert(LinkList &L,int i,ElemType e)
// 如果线性表L存在，将元素e插入到线性表L的第i个元素之前，返回OK；当插入位置不正确时，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
    if(!L) return INFEASIBLE;
    int len=ListLength(L);
    if(i<1||i>len+1) return ERROR;
    LinkList ll=L;
    while(--i) ll=ll->next;
    LinkList p=ll->next;
    LinkList temp=(LinkList)malloc(sizeof(LNode));
    temp->data=e;
    ll->next=temp;
    temp->next=p;
    return OK;
}


status ListDelete(LinkList &L,int i,ElemType &e)
// 如果线性表L存在，删除线性表L的第i个元素，并保存在e中，返回OK；当删除位置不正确时，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
    if(!L) return INFEASIBLE;
    int len=ListLength(L);
    if(i<1||i>len) return ERROR;
    LinkList ll=L;
    while(--i) ll=ll->next;
    LinkList p=ll->next;
    e=p->data;
    ll->next=ll->next->next;
    free(p);
    return OK;
}


status ListTraverse(LinkList L)
// 如果线性表L存在，依次显示线性表中的元素，每个元素间空一格，返回OK；如果线性表L不存在，返回INFEASIBLE。
{
    if(!L) return INFEASIBLE;
    while(L->next)
    {
        L=L->next;
        printf("%d ",L->data);
    }
    return OK;
}


status SaveList(LinkList L,char FileName[])
// 如果线性表L存在，将线性表L的的元素写到FileName文件中，返回OK，否则返回INFEASIBLE。
{
    if(!L) return INFEASIBLE;
    FILE* fp=fopen(FileName,"w");
    while(L->next){
        L=L->next
        fprintf(fp,"%d ",L->data);
    }
    fclose(fp);
    return OK;
}


status LoadList(LinkList &L,char FileName[])
// 如果线性表L不存在，将FileName文件中的数据读入到线性表L中，返回OK，否则返回INFEASIBLE。
{
    if(L!=NULL) return INFEASIBLE;
    FILE*fp=fopen(FileName,"r");
    L=(LinkList)malloc(sizeof(LNode));
    L->next=NULL;
    LinkList tail=L;
    int j;
    while(fscanf(fp,"%d",&j)!=EOF){
        LinkList p=(LinkList)malloc(sizeof(LNode));
        p->data=j;
        tail->next=p;
        tail=p;
    }
    tail->next=NULL;
    return OK;
}
