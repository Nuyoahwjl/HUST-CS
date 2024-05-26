#include "stdio.h"
#include "stdlib.h"
#define OK 1
#define INFEASIBLE -1
typedef int status;
typedef int ElemType; 
typedef struct LNode{  
      ElemType data;
      struct LNode *next;
}LNode,*LinkList;

status InitList(LinkList *L);

int main()
{
    LinkList L=NULL;
    InitList(&L);
    if(L==NULL) printf("no");
    else printf("yes");
    return 0;
}

status InitList(LinkList *L)
// 线性表L不存在，构造一个空的线性表，返回OK，否则返回INFEASIBLE。
{
    if(*L==NULL)
    {
        *L=(LinkList)malloc(sizeof(LNode));
        (*L)->next=NULL;
        return OK;
    }
    else return INFEASIBLE;
}


// #include <iostream>
// using namespace std;
// #define OK 1
// #define INFEASIBLE -1
// typedef int status;
// typedef int ElemType; 
// typedef struct LNode{  
//       ElemType data;
//       struct LNode *next;
// }LNode,*LinkList;

// status InitList(LinkList &L); // 引用参数

// int main()
// {
//     LinkList L=NULL;
//     InitList(L);
//     if(L==NULL) printf("no");
//     else printf("yes");
//     return 0;
// }

// status InitList(LinkList &L)
// // 线性表L不存在，构造一个空的线性表，返回OK，否则返回INFEASIBLE。
// {
//     if(L==NULL)
//     {
//         L=(LinkList)malloc(sizeof(LNode));
//         L->next=NULL;
//         return OK;
//     }
//     else return INFEASIBLE;
// }