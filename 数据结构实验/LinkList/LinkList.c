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
{
    #include "def.h"
    #include "stu.h"
    int main() {
        LinkList L;
        int i,j;
        scanf("%d",&i);
        if (!i) { 
            L = NULL;
            j = InitList(L);
            if (L==NULL) printf("可能没有正确分配表头节点空间");
            if (L->next!=NULL) printf("表头节点可能没有正确初始化");
            if (j==OK) printf("OK");
        }
        else {
            L=(LinkList)malloc(sizeof(LNode));
            j=InitList(L);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对已经存在的线性表初始化");
            free(L);
        }
        return 1;
    }
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
{
    #include "def.h"
    #include "string.h"
    #include "ctype.h"
    #define free free0
    #include "stu.h"
    #undef free
    struct ptr{
        void *pused[100],*pfree[100];
        int len_used,len_free;
        } pm;
    void free0(void *p)
    {
        pm.pfree[pm.len_free++]=p;
        memset(p,0,sizeof(LNode));
        free(p);
    }
    
    int main()
    {
        LinkList L,tail;
        int f,i,j;
        scanf("%d",&f);
        if (f==0)
        {
            L=NULL;
            j=DestroyList(L);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("不能对不存在的线性表进行销毁操作！");
        }
        else
        {
            pm.pused[pm.len_used++]=tail=L=(LinkList) malloc(sizeof(LNode));
            scanf("%d",&f);
            while (f)
            {
                pm.pused[pm.len_used++]=tail->next=(LNode*) malloc(sizeof(LNode));
                tail=tail->next;
                tail->data=f;
                scanf("%d",&f);
            }
            tail->next=NULL;
            j=DestroyList(L);
            if (j==OK && !L)
            {
                for(i=0;i<pm.len_used;i++)
                {
                    for(j=0;j<pm.len_free;j++)
                        if (pm.pused[i]==pm.pfree[j])
                            break;
                    if (j>=pm.len_free)
                    {
                        printf("未正确释放数据元素空间");
                        return 1;
                    }
                }
                printf("OK");
            }
            else printf("ERROR!");
        }
        return 1;
    }
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
{
    #include "def.h"
    #include "string.h"
    #include "ctype.h"
    #define free free0
    #include "stu.h"
    #undef free
    struct ptr{
        void *pused[100],*pfree[100];
        int len_used,len_free;
        } pm;
    void free0(void *p)
    {
        pm.pfree[pm.len_free++]=p;
        free(p);
    }


    int main()
    {
        LinkList L,tail;
        int f,i,j;
        scanf("%d",&f);
        if (f==0)
        {
            L=NULL;
            j=ClearList(L);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("不能对不存在的线性表进行销毁操作！");
        }
        else
        {
            pm.pused[pm.len_used++]=tail=L=(LinkList) malloc(sizeof(LNode));
            scanf("%d",&f);
            while (f)
            {
                pm.pused[pm.len_used++]=tail->next=(LNode*) malloc(sizeof(LNode));
                tail=tail->next;
                tail->data=f;
                scanf("%d",&f);
            }
            tail->next=NULL;
            j=ClearList(L);
            if (j==OK && L && !L->next && pm.len_used==pm.len_free+1)
            {
                for(i=1;i<pm.len_used;i++)
                {
                    for(j=0;j<pm.len_free;j++)
                        if (pm.pused[i]==pm.pfree[j])
                            break;
                    if (j>=pm.len_free)
                    {
                        printf("未正确释放数据元素空间");
                        return 1;
                    }
                }
                printf("OK");
            }
            else printf("ERROR!");
        }
        return 1;
    }
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
{
    #include "def.h"
    #include "stu.h"
    int main() {
        LinkList L;
        int i,j;
        scanf("%d",&i);
        if (i==2) { 
            L=(LinkList) malloc(sizeof(LNode));
            L->next=NULL;
            j=ListEmpty(L);
            if (j==OK) printf("TRUE");
            else printf("未正确判空");
            free(L);
        }
        else if(i==1) {
            L=(LinkList) malloc(sizeof(LNode));
            L->next=(LNode*) malloc(sizeof(LNode));
            L->next->next=NULL;
            j=ListEmpty(L);
            if (j==ERROR) printf("FALSE");
            else printf("未正确判空");
            free(L->next);
            free(L);
        }
        else {
            L=NULL;
            j=ListEmpty(L);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表判空");
        }
        return 1;
    }
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
{
    #include "def.h"
    #include "stu.h"
    int main() {
        LinkList L;
        int f,i,j;
        scanf("%d",&f);
        if (!f) {
            L=NULL;
            j=ListLength(L);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表求表长");
        }
        else {
            L=(LinkList) malloc(sizeof(LNode));
            L->next=NULL;
            LNode *s,*r=L;
            scanf("%d",&i);
            while (i) {
                s=(LNode*) malloc(sizeof(LNode));
                s->data=i;
                r->next=s;
                r=s;
                scanf("%d",&i);
            }
            r->next=NULL;
            j=ListLength(L);
            printf("%d", j);
        }
        return 1;
    }
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
{
    #include "def.h"
    #include "stu.h"
    int main() {
        LinkList L;
        int f,i,j,e;
        scanf("%d",&f);
        if (!f) {
            L=NULL;
            j=GetElem(L,2,e);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表求表长");
        }
        else {
            L=(LinkList) malloc(sizeof(LNode));
            L->next=NULL;
            LNode *s,*r=L;
            scanf("%d",&i);
            while (i) {
                s=(LNode*) malloc(sizeof(LNode));
                s->data=i;
                r->next=s;
                r=s;
                scanf("%d",&i);
            }
            r->next=NULL;
            scanf("%d",&i);
            j=GetElem(L,i,e);
            if(j==OK) printf("OK\n%d",e);
            if(j==ERROR) printf("ERROR");
        }
        return 1;
    }
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
{
    #include "def.h"
    #include "stu.h"
    int main() {
        LinkList L;
        int f,i,j,e;
        scanf("%d",&f);
        if (!f) {
            L=NULL;
            j=LocateElem(L,e);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表求表长");
        }
        else {
            L=(LinkList) malloc(sizeof(LNode));
            L->next=NULL;
            LNode *s,*r=L;
            scanf("%d",&i);
            while (i) {
                s=(LNode*) malloc(sizeof(LNode));
                s->data=i;
                r->next=s;
                r=s;
                scanf("%d",&i);
            }
            r->next=NULL;
            scanf("%d",&e);
            j=LocateElem(L,e);
            if(j==ERROR) printf("ERROR");
            else printf("%d",j);
        }
        return 1;
    }
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
{
    #include "def.h"
    #include "stu.h"
    int main() {
        LinkList L;
        int f,i,j,e,pre;
        scanf("%d",&f);
        if (!f) {
            L=NULL;
            j=PriorElem(L,e,pre);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表求表长");
        }
        else {
            L=(LinkList) malloc(sizeof(LNode));
            L->next=NULL;
            LNode *s,*r=L;
            scanf("%d",&i);
            while (i) {
                s=(LNode*) malloc(sizeof(LNode));
                s->data=i;
                r->next=s;
                r=s;
                scanf("%d",&i);
            }
            r->next=NULL;
            scanf("%d",&e);
            j=PriorElem(L,e,pre);
            if(j==ERROR) printf("ERROR");
            if(j==OK) printf("OK\n%d",pre);
        }
        return 1;
    }
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
{
    #include "def.h"
    #include "stu.h"
    int main() {
        LinkList L;
        int f,i,j,e,next;
        scanf("%d",&f);
        if (!f) {
            L=NULL;
            j=NextElem(L,e,next); 
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表求表长");
        }
        else {
            L=(LinkList) malloc(sizeof(LNode));
            L->next=NULL;
            LNode *s,*r=L;
            scanf("%d",&i);
            while (i) {
                s=(LNode*) malloc(sizeof(LNode));
                s->data=i;
                r->next=s;
                r=s;
                scanf("%d",&i);
            }
            r->next=NULL;
            scanf("%d",&e);
            j=NextElem(L,e,next);
            if(j==ERROR) printf("ERROR");
            if(j==OK) printf("OK\n%d",next);
        }
        return 1;
    }
}

#include "../step05/stu.h"
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
{
    #include "def.h"
    #include "stu.h"
    int main() {
        LinkList L;
        int f,i,j,e,next;
        scanf("%d",&f);
        if (!f) {
            L=NULL;
            j=ListInsert(L,1,1);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("不能对不存在的线性表进行插入操作！");
        }
        else {
            L=(LinkList) malloc(sizeof(LNode));
            L->next=NULL;
            LNode *s,*r=L;
            scanf("%d",&i);
            while (i) {
                s=(LNode*) malloc(sizeof(LNode));
                s->data=i;
                r->next=s;
                r=s;
                scanf("%d",&i);
            }
            r->next=NULL;
            scanf("%d%d",&i,&e);
            j=ListInsert(L,i,e);
            printf("%s\n", j==OK? "OK" : j==ERROR? "ERROR" : "");
            for(s=L->next;s;s=s->next)
                printf("%d ",s->data);
        }
        return 1;
    }
}

#include "../step05/stu.h"
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
{
    #include "def.h"
    #include "ctype.h"
    #include "string.h"
    #define free free0
    #include "stu.h"
    #undef free

    struct ptr{
        void *pused[100],*pfree[100];
        int len_used,len_free;
        } pm;
    void free0(void *p)
    {
        pm.pfree[pm.len_free++]=p;
        free(p);
    }

    int main() {
        LinkList L;
        int f,i,j,e;
        scanf("%d",&f);
        if (!f) {
            L=NULL;
            j=ListDelete(L,1,e);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("不能对不存在的线性表进行删除操作！");
        }
        else {
            L=(LinkList) malloc(sizeof(LNode));
            L->next=NULL;
            LNode *s,*r=L;
            scanf("%d",&i);
            while (i) {
                pm.pused[pm.len_used++]=s=(LNode*) malloc(sizeof(LNode));
                s->data=i;
                r->next=s;
                r=s;
                scanf("%d",&i);
            }
            r->next=NULL;
            scanf("%d",&i);
            j=ListDelete(L,i,e);
            if(j==ERROR) printf("ERROR\n");
            if(j==OK) {
                for(i=0;i<pm.len_used;i++)
                    if (pm.pfree[0]==pm.pused[i]) break;
                if (pm.len_free!=1 || i>=pm.len_used)
                {
                    printf("未正确释放数据元素空间");
                    return 1;
                }
                printf("OK\n%d\n",e);
            }
            for(s=L->next;s;s=s->next)
                printf("%d ",s->data);
        }
        return 1;
    }
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
{
    #include "def.h"
    #include "stu.h"

    int main() {
        LinkList L;
        int f,i,j,e;
        scanf("%d",&f);
        if (!f) {
            L=NULL;
            j=ListTraverse(L);
            if (j==INFEASIBLE) printf("INFEASIBLE");
            else printf("可能会对不存在的线性表进行遍历操作！");
        }
        else {
            L=(LinkList) malloc(sizeof(LNode));
            L->next=NULL;
            LNode *s,*r=L;
            scanf("%d",&i);
            while (i) {
                s=(LNode*) malloc(sizeof(LNode));
                s->data=i;
                r->next=s;
                r=s;
                scanf("%d",&i);
            }
            r->next=NULL;
            j=ListTraverse(L);
            if(j==OK && L->next==NULL) printf("空线性表");
        }
        return 1;
    }
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
{
    #include "def.h"
    #include "stu.h"
    #include "string.h"

    int main() {
        LinkList L;
        int f,i=0,j,e;
        FILE *fp;
        char FileName[30];
        strcpy(FileName,"src/step13/list.dat");
        scanf("%d",&f);
        if (!f) {
            L=NULL;
            j=SaveList(L,"src/step13/list.dat");
            if (j==INFEASIBLE) printf("INFEASIBLE\n");
            else printf("不能对不存在的线性表进行写文件操作！\n");
            
            L=(LinkList) malloc(sizeof(LNode));
            L->next=NULL;
            j=LoadList(L,"");
            if (j==INFEASIBLE) printf("INFEASIBLE\n");
            else printf("不能对已存在的线性表进行写操作！否则会覆盖原数据，造成数据丢失\n");
        }
        else {
            L=(LinkList) malloc(sizeof(LNode));
            L->next=NULL;
            LNode *s,*r=L;
            scanf("%d",&i);
            while (i) {
                s=(LNode*) malloc(sizeof(LNode));
                s->data=i;
                r->next=s;
                r=s;
                scanf("%d",&i);
            }
            r->next=NULL;
            j=SaveList(L,"src/step13/list.dat");
            if(j==OK) printf("OK\n");
            while(L)
            {
                s = L->next;
                L->data=0;
                free(L);
                L = s;
            }
            j=LoadList(L,"src/step13/list.dat");
            if(j==OK) printf("OK\n");
            for(s=L->next;s;s=s->next)
                printf("%d ",s->data);
        }
        return 1;
    }
}