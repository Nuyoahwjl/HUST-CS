/* 引用头文件 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* 定义常量 */
#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASIBLE -1
#define OVERFLOW -2
#define LIST_INIT_SIZE 100  // 初始链表长度
#define LISTINCREMENT 10    // 每次新增长度
#define MAX_LIST_NUM 10     // 链表数量最大值
#define MAX_NAME_LENGTH 30  // 每个线性表名称长度最大值
#define FileName "data.txt" // 文件名
/* 定义数据元素类型 */
typedef int ElemType;
typedef int status;
/* 单链表(链式结构)结点的定义 */
typedef struct LNode
{
    ElemType data;
    struct LNode *next;
} LNode, *LinkList;
/* 链表的集合类型定义 */
typedef struct
{
    struct
    {
        char name[MAX_NAME_LENGTH]; // 链表名称
        LinkList L;                 // 指向链表表头的指针
    } elem[MAX_LIST_NUM];
    int length; // 当前链表数量
} LISTS;
LISTS Lists; // 链表集合实例化为Lists对象(全局变量)


/* 函数声明 */
status checkList(LinkList L); // 检查该链表是否初始化
void printMenu1(void); // 链表集合的管理菜单
void printMenu2(void); // 单个链表的操作菜单
status AddList(LISTS *LL,char *name); // 新增一个链表
status RemoveList(LISTS *LL,char *name); // 移除一个链表
status ShowAllLists(LISTS Lists); // 显示所有链表
int SelectList(LISTS Lists,char *name); // 选择一个链表并进入子菜单
void main2(LinkList *L,int loc); // 操作单个链表的子页面
status InitList(LinkList *L); // 初始化链表

/* 主函数 */
int main()
{
    printMenu1();
    Lists.length=0;
    int op=1;
    while(op)
    {
        // 提示用户选择操作
        printf("\n|-------------------------------------------------------------|\n");
        printf(  "|-------Please Choose Your Operation from Options above-------|\n");
        printf(  "|-------------------------------------------------------------|\n\n");
        scanf("%d", &op); // 读取用户输入
        system("cls");    // 清屏
        printMenu1();
        switch(op)
        {
            case 1: // 创建一个链表
            {
                printf("|-------------------------------------------------------------|\n"
                       "|---------You can create a total of %d linear tables.---------|\n",
                       MAX_LIST_NUM); // 可创建线性表总数
                // 当前创建线性表的数量
                if (Lists.length > 1)
                    printf("|--------Currently, %d linear tables have been created.--------|\n", Lists.length);
                else
                    printf("|---------Currently, %d linear table has been created.---------|\n", Lists.length);
                printf("|-------------------------------------------------------------|\n\n");
                char add_name[MAX_NAME_LENGTH];
                printf("Please enter the name of the linear table you want to add : \n");
                scanf("%s", add_name);
                if (AddList(&Lists, add_name) == OK) // 创建成功
                    printf("The linear table (name: %s) is created!\n", add_name);
                else
                    printf("The linear table already exists.\n");
                break;
            }
            case 2: // 删除一个链表
            {
                char remove_name[MAX_NAME_LENGTH];
                printf("Please enter the name of the linear table you want to delete : \n");
                scanf("%s", remove_name);
                if (RemoveList(&Lists, remove_name) == OK) // 删除成功
                    printf("The linear table (name: %s) is deleted!\n", remove_name);
                else
                    printf("The linear table does not exist.\n");
                break;
            }
            case 3: // 显示所有链表
            {
                ShowAllLists(Lists);
                break;
            } 
            case 4: // 选择一个链表并进入子菜单
            {
                char select_name[MAX_NAME_LENGTH];
                printf("Please enter the name of the linear table you want to select : \n");
                scanf("%s", select_name);
                int loc = SelectList(Lists, select_name); // 所选链表在链表集合的位置
                if (loc != -1) // 所选链表存在
                    {
                        main2(&Lists.elem[loc].L,loc);  // 跳转至子页面
                        printMenu1();
                    }
                else
                    printf("The linear table does not exist.\n");
                break;
            }
            case 0:
                break;
            default:
                printf("The feature number is incorrect.\n"); // 功能选项错误
        }
    }
    printf("Welcome to use this system next time!\n");
    return 0;
}

/* 操作单个链表的子页面 */
void main2(LinkList *L,int loc)
{
    system("cls");
    printMenu2();
    int op=1;
    while(op)
    {
        if(*L==NULL) printf("Don't forget to initialize current List.\n");
        // 提示用户选择操作
        printf("|-------------------------------------------------------------|\n");
        printf("|-------Please Choose Your Operation from Options above-------|\n");
        printf("|-------------------------------------------------------------|\n\n");
        scanf("%d", &op); // 读取用户输入
        system("cls");    // 清屏
        printMenu2();
        switch(op)
        {
            case 1:
                if(InitList(L)==OK) printf("Successfully initialized!\n");
                else printf("You've initialized current List!\n");
                break;
            case 0:
                system("cls");
                return;

        }
    }
}

/* 检查当前链表是否初始化 */
status checkList(LinkList L)
{
    if (L == NULL)
    {
        printf("You need to initialize first.\n");
        return FALSE;
    }
    else
        return TRUE;
}

/* 链表集合的管理菜单 */
void printMenu1()
{
    printf("|=================Menu for multiple LinkLists=================|\n");
    printf("|-------------------------------------------------------------|\n");
    printf("|                 1.    Create a LinkList                     |\n"); // 打印操作1的描述
    printf("|                 2.    Delete a LinkList                     |\n"); // 打印操作2的描述
    printf("|                 3.    Show all LinkLists                    |\n"); // 打印操作3的描述
    printf("|                 4. Select a single LinkList                 |\n"); // 打印操作4的描述
    printf("|                 0.          EXIT                            |\n"); // 打印操作0的描述
    printf("|=============================================================|\n\n");
}

/* 单个链表的操作菜单 */
void printMenu2()
{
    printf("|===================Menu for single LinkList==================|\n");
    printf("|-------------------------------------------------------------|\n");
    printf("|      1.  Init current List      2.  Destroy Current List    |\n");
    printf("|      3.  Clear current List     4.  Empty or Not            |\n");
    printf("|      5.  Show List Length       6.  Get Element             |\n");
    printf("|      7.  Locate Elem            8.  Get Prior Element       |\n");
    printf("|      9.  Get Next Element       10. Insert Element          |\n");
    printf("|                        0.  EXIT                             |\n");
    printf("|-------------------------------------------------------------|\n\n");
}

/* 新增一个链表 */
status AddList(LISTS *LL,char *name)
{
    // 查询是否存在同名链表
    for (int i = 0; i < LL->length; i++)
        if (strcmp(LL->elem[i].name, name) == 0)
            return INFEASIBLE;
    // 不存在，可以新增
    strcpy(LL->elem[LL->length].name, name); // 给链表名称赋值
    LL->elem[LL->length].L = NULL;           // 头指针初始化为空(未初始化状态)
    LL->length++;                            // 长度加一
    return OK;
}

/* 移除一个链表 */
status RemoveList(LISTS *LL,char *name)
{
    // 查询是否存在
    for (int loc = 0; loc < LL->length; loc++)
    {
        if (strcmp(LL->elem[loc].name, name) == 0) // 存在
        {
            // DestroyList(LL->elem[loc].L); // 先销毁本身空间
            for (int i = loc; i < LL->length - 1; i++)
                LL->elem[i] = LL->elem[i + 1];
            LL->length--;
            return OK; // 移除成功
        }
    }
    return ERROR; // 不存在，返回错误
}

/* 显示所有链表 */
status ShowAllLists(LISTS Lists)
{
    for (int i = 0; i < Lists.length; i++)
    {
        printf("num%d\n", i);                    // 打印序号
        printf("name:%s\n", Lists.elem[i].name); // 打印名称
        printf("elements:");
        if (Lists.elem[i].L == NULL) // 未初始化
            printf("You haven't initialized.");
        else if (Lists.elem[i].L->next == NULL) // 无元素
            printf("none");
        // else
        //     ListTraverse(Lists.elem[i].L); //遍历
        printf("\n");
    }
    return OK;
}

/* 选择一个链表并进入子菜单 */
int SelectList(LISTS Lists, char *name)
{
    for (int i = 0; i < Lists.length; i++)
        if (strcmp(Lists.elem[i].name, name) == 0) // 所选链表存在
            return i; // 返回位置
    return -1;
}

/* 初始化链表 */
status InitList(LinkList *L)
{
    if(*L==NULL)
    {
        *L=(LinkList)malloc(sizeof(LNode));
        (*L)->next=NULL;
        return OK;
    }
    else return INFEASIBLE;
}