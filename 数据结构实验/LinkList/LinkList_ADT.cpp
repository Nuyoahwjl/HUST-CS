// 用C++来写，不用考虑指针和参数回传，直接使用引用参数
// case语句中定义变量时不能赋初值，编译器会报错

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
status checkList(LinkList L);                             // 判断链表是否初始化
void printMenu1(void);                                    // 链表集合的管理菜单
void printMenu2(void);                                    // 单个链表的操作菜单
status AddList(LISTS &LL, char *name);                    // 新增一个链表
status RemoveList(LISTS &LL, char *name);                 // 移除一个链表
status ShowAllLists(LISTS Lists);                         // 显示所有链表
int SelectList(LISTS Lists, char *name);                  // 选择一个链表并进入子菜单
void main2(LinkList &L);                                  // 操作单个链表的子页面
status InitList(LinkList &L);                             // 初始化链表
status DestroyList(LinkList &L);                          // 销毁链表
status ClearList(LinkList &L);                            // 清空链表元素
status ListEmpty(LinkList L);                             // 判断链表是否为空
int ListLength(LinkList L);                               // 链表长度
status GetElem(LinkList L, int i, ElemType &e);           // 获取第i个元素
status LocateElem(LinkList L, ElemType e);                // 获取元素位置
status PriorElem(LinkList L, ElemType e, ElemType &pre);  // 获取前驱元素
status NextElem(LinkList L, ElemType e, ElemType &next);  // 获取后继元素
status ListInsert(LinkList &L, int i, ElemType e);        // 指定位置前插入元素
status ListDelete(LinkList &L, int i, ElemType &e);       // 删除指定位置元素
void visit(ElemType item);                                // 用于遍历时输出
status ListTraverse(LinkList L, void (*visit)(ElemType)); // 显示当前链表所有元素
void ListReverse(LinkList &L);                            // 逆置当前链表
void RemoveNthFromEnd(LinkList &L, int n);                // 删除倒数第n个元素
void SortList(LinkList &L);                               // 给当前链表排序
void SaveData(LISTS Lists);                               // 保存到文件
void LoadData(LISTS &Lists);                              // 从文件中加载

/* 主函数 */
int main()
{
    printMenu1(); // 打印主菜单
    Lists.length = 0;
    int op = 1;
    while (op)
    {
        // 提示用户选择操作
        printf("\n|-------------------------------------------------------------|\n");
        printf("|-------Please Choose Your Operation from Options above-------|\n");
        printf("|-------------------------------------------------------------|\n\n");
        scanf("%d", &op); // 读取用户输入
        system("cls");    // 清屏
        printMenu1();
        int save_flag = 0;   // 保存到文件时是否弹出提示的标志
        int save_option = 0; // 保存文件时用户是否确认
        int load_option = 1; // 从文件中加载时用户是否确认
        switch (op)
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
            char add_name[MAX_NAME_LENGTH]; // 新增链表的名称
            printf("Please enter the name of the linear table you want to add : \n");
            scanf("%s", add_name);
            if (AddList(Lists, add_name) == OK) // 创建成功
                printf("The linear table (name: %s) is created!\n", add_name);
            else // 已经存在
                printf("The linear table already exists.\n");
            break;
        }
        case 2: // 删除一个链表
        {
            char remove_name[MAX_NAME_LENGTH]; // 删除链表的名称
            printf("Please enter the name of the linear table you want to delete : \n");
            scanf("%s", remove_name);
            if (RemoveList(Lists, remove_name) == OK) // 删除成功
                printf("The linear table (name: %s) is deleted!\n", remove_name);
            else // 不存在
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
            char select_name[MAX_NAME_LENGTH]; // 选择的链表名称
            printf("Please enter the name of the linear table you want to select : \n");
            scanf("%s", select_name);
            int loc = SelectList(Lists, select_name); // 所选链表在链表集合的位置
            if (loc != -1)                            // 所选链表存在
            {
                main2(Lists.elem[loc].L); // 跳转至子页面
                printMenu1();             // 从子页面跳回时再次打印主菜单
            }
            else // 所选链表不存在
                printf("The linear table does not exist.\n");
            break;
        }
        case 5: // 保存到文件
            for (int i = 0; i < Lists.length; i++)
            {
                if (Lists.elem[i].L == NULL) // 当前Lists是有未初始化的链表
                {
                    save_flag = 1;
                    break;
                }
            }
            if (save_flag) // 当前Lists是有未初始化的链表，弹出确认提示
            {
                printf("The lists that are not initialized will not be saved.\n");
                printf("confirm:1  cancel:0\n"); // 未初始化的文件不会被保存
                scanf("%d", &save_option);
                if (save_option) // 确认保存到文件
                {
                    SaveData(Lists);
                    printf("Successfully Saved.\n"); // 保存成功
                }
            }
            else
            {
                SaveData(Lists);
                printf("Successfully Saved.\n"); // 保存成功
            }
            break;
        case 6:               // 从文件中加载
            if (Lists.length) // 当前Lists非空，弹出确认提示
            {
                printf("Are you sure you want to load from the file?\n");
                printf("The data that is not currently saved will be gone.\n");
                printf("confirm:1  cancel:0\n"); // 从文件中读取会覆盖当前Lists
                scanf("%d", &load_option);
            }
            if (load_option) // 确认从文件中加载
            {
                LoadData(Lists);
                printf("Successfully Loaded.\n");                                // 加载成功
                printf("Now you can enter 3 to query all lists in the file.\n"); // 输入3显示所有链表
            }
            break;
        case 7: // 查找链表位置
            char search_name[MAX_NAME_LENGTH];
            printf("Please enter the name of the list you want to query:\n");
            scanf("%s", search_name);
            if (SelectList(Lists, search_name) != -1)
                printf("The location of the list is %d.\n", SelectList(Lists, search_name) + 1);
            else
                printf("There is no list named %s.\n", search_name);
            break;
        case 0: // 退出
            break;
        default:
            printf("The feature number is incorrect.\n"); // 功能选项错误
        }
    }
    printf("Welcome to use this system next time!\n"); // 欢迎下次使用
    return 0;
}

/* 操作单个链表的子页面 */
void main2(LinkList &L)
{
    system("cls"); // 清屏
    printMenu2();  // 打印子菜单
    int op = 1;
    int flag = 1; // 初次进入该页面，提醒用户初始化
    while (op)
    {
        if (L == NULL && flag)
        {
            printf("Don't forget to initialize current List.\n");
            flag = 0;
        }
        // 提示用户选择操作
        printf("\n|-------------------------------------------------------------|\n");
        printf("|-------Please Choose Your Operation from Options above-------|\n");
        printf("|-------------------------------------------------------------|\n\n");
        scanf("%d", &op); // 读取用户输入
        system("cls");    // 清屏
        printMenu2();     // 打印子菜单
        switch (op)
        {
        case 1:                    // 初始化链表
            if (InitList(L) == OK) // 初始化成功
                printf("Successfully initialized!\n");
            else // 已经初始化
                printf("You've initialized current List!\n");
            break;
        case 2:               // 销毁链表
            if (checkList(L)) // 已初始化
            {
                if (DestroyList(L) == OK) // 销毁成功
                    printf("Successfully destroyed!\n");
            }
            break;
        case 3:               // 清空链表元素
            if (checkList(L)) // 已初始化
            {
                if (ClearList(L) == OK) // 清空成功
                    printf("Successfully cleared!\n");
            }
            break;
        case 4:               // 判断链表是否为空
            if (checkList(L)) // 已初始化
            {
                if (ListEmpty(L) == TRUE) // 为空
                    printf("The List is empty.\n");
                else if (ListEmpty(L) == FALSE) // 不为空
                    printf("The List is not empty.\n");
            }
            break;
        case 5:               // 链表长度
            if (checkList(L)) // 已初始化
            {
                if (ListLength(L) != INFEASIBLE)
                    printf("The length of the List is : %d.\n", ListLength(L));
            }
            break;
        case 6:               // 获取第i个元素
            if (checkList(L)) // 已初始化
            {
                if (ListLength(L)) // 长度不为零
                {
                    printf("Please enter the position (between 1 to %d) you want to query:\n", ListLength(L));
                    int queryPosition; // 查询的位置
                    scanf("%d", &queryPosition);
                    ElemType queryResult;                             // 查询的结果
                    if (GetElem(L, queryPosition, queryResult) == OK) // 位置合法
                        printf("The element is %d.\n", queryResult);
                    else if (GetElem(L, queryPosition, queryResult) == ERROR) // 位置不合法
                        printf("The position is illegal.\n");
                }
                else // 长度为零
                    printf("There is no element.\n");
            }
            break;
        case 7:               // 获取元素位置
            if (checkList(L)) // 已初始化
            {
                if (ListLength(L)) // 长度不为零
                {
                    printf("Please enter the element you want to locate.\n");
                    ElemType queryElem_locate; // 查询的元素
                    scanf("%d", &queryElem_locate);
                    if (LocateElem(L, queryElem_locate) != ERROR) // 存在
                        printf("The position of %d is %d.\n", queryElem_locate, LocateElem(L, queryElem_locate));
                    else
                        printf("The element does not exist.\n"); // 不存在
                }
                else // 长度为零
                    printf("There is no element.\n");
            }
            break;
        case 8:               // 获取前驱元素
            if (checkList(L)) // 已初始化
            {
                if (ListLength(L)) // 长度不为零
                {
                    printf("Please enter the element you want to query:\n");
                    ElemType queryElem_prior, pre; // 查询的元素及前驱元素
                    scanf("%d", &queryElem_prior);
                    if (PriorElem(L, queryElem_prior, pre) == OK) // 位置合法且元素存在
                        printf("The prior element of %d is %d.\n", queryElem_prior, pre);
                    else if (PriorElem(L, queryElem_prior, pre) == ERROR) // 位置不合法或元素不存在
                        printf("failed to find.\n");
                }
                else // 长度为零
                    printf("There is no element.\n");
            }
            break;
        case 9:               // 获取后继元素
            if (checkList(L)) // 已初始化
            {
                if (ListLength(L)) // 长度不为零
                {
                    printf("Please enter the element you want to query:\n");
                    ElemType queryElem_next, next; // 查询的元素及后继元素
                    scanf("%d", &queryElem_next);
                    if (NextElem(L, queryElem_next, next) == OK) // 位置合法且元素存在
                        printf("The next element of %d is %d.\n", queryElem_next, next);
                    else if (NextElem(L, queryElem_next, next) == ERROR) // 位置不合法或元素不存在
                        printf("failed to find.\n");
                }
                else // 长度为零
                    printf("There is no element.\n");
            }
            break;
        case 10:              // 指定位置前插入元素
            if (checkList(L)) // 已初始化
            {
                printf("Position: (between 1 to %d)\n", ListLength(L) + 1); // 可供选择的位置
                printf("Please enter the position and the element you want to insert:(spaced by space)\n");
                int insertPosition;  // 插入的位置
                ElemType insertElem; // 插入的元素
                scanf("%d %d", &insertPosition, &insertElem);
                if (ListInsert(L, insertPosition, insertElem) == OK) // 插入成功
                    printf("Successfully inserted.\n");
                else if (ListInsert(L, insertPosition, insertElem) == ERROR) // 位置不合法
                    printf("The position is illegal.\n");
            }
            break;
        case 11:              // 删除指定位置元素
            if (checkList(L)) // 已初始化
            {
                if (ListLength(L)) // 长度不为零
                {
                    printf("Position: (between 1 to %d)\n", ListLength(L)); // 可供选择的位置
                    printf("Please enter the position you want to delete:\n");
                    int deletePosition;  // 删除的位置
                    ElemType deleteElem; // 删除的元素
                    scanf("%d", &deletePosition);
                    if (ListDelete(L, deletePosition, deleteElem) == OK) // 删除成功
                        printf("Delete %d in position %d.\n", deleteElem, deletePosition);
                    else if (ListDelete(L, deletePosition, deleteElem) == ERROR) // 位置不合法
                        printf("The position is illegal.\n");
                }
                else // 长度为零
                    printf("There is no element.\n");
            }
            break;
        case 12:              // 显示当前链表所有元素
            if (checkList(L)) // 已初始化
            {
                if (ListLength(L)) // 长度不为零
                {
                    if (ListTraverse(L, visit) == OK) // 成功遍历
                        printf("\nSuccessfully traveled all elements.\n");
                }
                else // 长度为零
                    printf("There is no element.\n");
            }
            break;
        case 13:              // 逆置当前链表
            if (checkList(L)) // 已初始化
            {
                if (ListLength(L)) // 长度不为零
                {
                    ListReverse(L); // 逆置
                    printf("Successfully reversed.\n");
                }
                else // 长度为零
                    printf("There is no element.\n");
            }
            break;
        case 14:              // 删除倒数第n个元素
            if (checkList(L)) // 已初始化
            {
                if (ListLength(L)) // 长度不为零
                {
                    int n; // 删除的位置
                    printf("Which element do you want to remove from end?\n");
                    scanf("%d", &n);
                    RemoveNthFromEnd(L, n); // 删除
                }
                else // 长度为零
                    printf("There is no element.\n");
            }
            break;
        case 15:              // 给当前链表排序
            if (checkList(L)) // 已初始化
            {
                if (ListLength(L)) // 长度不为零
                {
                    SortList(L); // 排序
                    printf("Successfully sorted.\n");
                }
                else // 长度为零
                    printf("There is no element.\n");
            }
            break;
        case 0:
            system("cls");
            return;
        default:
            printf("The feature number is incorrect.\n");
        } // end of switch
    } // end of while
}

/* 判断链表是否初始化 */
status checkList(LinkList L)
{
    if (L == NULL) // L未分配空间
    {
        printf("You haven't initialized.\n");
        return FALSE;
    }
    return TRUE;
}

/* 链表集合的管理菜单 */
void printMenu1()
{
    printf("|=================Menu for multiple LinkLists=================|\n");
    printf("|-------------------------------------------------------------|\n");
    printf("|                 1.    Create a LinkList                     |\n");
    printf("|                 2.    Delete a LinkList                     |\n");
    printf("|                 3.    Show all LinkLists                    |\n");
    printf("|                 4. Select a single LinkList                 |\n");
    printf("|                 5.  Save All Data To File                   |\n");
    printf("|                 6. Load All Data From File                  |\n");
    printf("|                 7.    Search a LinkList                     |\n");
    printf("|                 0.          EXIT                            |\n");
    printf("|=============================================================|\n\n");
}

/* 单个链表的操作菜单 */
void printMenu2()
{
    printf("|===================Menu for single LinkList==================|\n");
    printf("|-------------------------------------------------------------|\n");
    printf("|      1.  Init Current List      2.  Destroy Current List    |\n");
    printf("|      3.  Clear Current List     4.  Empty or Not            |\n");
    printf("|      5.  Show List Length       6.  Get Element             |\n");
    printf("|      7.  Locate Element         8.  Get Prior Element       |\n");
    printf("|      9.  Get Next Element       10. Insert Element          |\n");
    printf("|      11. Delete Element         12. Show All Elements       |\n");
    printf("|      13. Reverse Current List   14. Remove From End         |\n");
    printf("|      15. Sort Current List      0. EXIT                     |\n");
    printf("|-------------------------------------------------------------|\n\n");
}

/* 新增一个链表 */
status AddList(LISTS &LL, char *name)
{
    // 查询是否存在同名链表
    for (int i = 0; i < LL.length; i++)
        if (strcmp(LL.elem[i].name, name) == 0)
            return INFEASIBLE;
    // 不存在，可以新增
    strcpy(LL.elem[LL.length].name, name); // 给链表名称赋值
    LL.elem[LL.length].L = NULL;           // 头指针初始化为空(未初始化状态)
    LL.length++;                           // 长度加一
    return OK;
}

/* 移除一个链表 */
status RemoveList(LISTS &LL, char *name)
{
    // 查询是否存在
    for (int loc = 0; loc < LL.length; loc++)
    {
        if (strcmp(LL.elem[loc].name, name) == 0) // 存在
        {
            DestroyList(LL.elem[loc].L); // 先销毁本身空间
            for (int i = loc; i < LL.length - 1; i++)
                LL.elem[i] = LL.elem[i + 1]; // 前移
            LL.length--;
            return OK; // 移除成功
        }
    }
    return ERROR; // 不存在，返回错误
}

/* 显示所有链表 */
status ShowAllLists(LISTS Lists)
{
    if (Lists.length == 0) // 无链表
        printf("There are no linear tables.\n");
    for (int i = 0; i < Lists.length; i++)
    {
        printf("No.%d\n", i + 1);                // 打印序号
        printf("name:%s\n", Lists.elem[i].name); // 打印名称
        printf("elements:");
        if (Lists.elem[i].L == NULL) // 未初始化
            printf("You haven't initialized.");
        else if (Lists.elem[i].L->next == NULL) // 无元素
            printf("none");
        else
            ListTraverse(Lists.elem[i].L, visit); // 遍历
        printf("\n");
    }
    return OK;
}

/* 选择一个链表并进入子菜单 */
int SelectList(LISTS Lists, char *name)
{
    for (int i = 0; i < Lists.length; i++)
        if (strcmp(Lists.elem[i].name, name) == 0) // 所选链表存在
            return i;                              // 返回其在链表集合中的位置
    return -1;
}

/* 初始化链表 */
status InitList(LinkList &L)
{
    if (L == NULL) // 未初始化
    {
        L = (LinkList)malloc(sizeof(LNode)); // 分配空间
        L->next = NULL;                      // 无元素，后继为空
        return OK;
    }
    else // 已初始化
        return INFEASIBLE;
}

/* 销毁链表 */
status DestroyList(LinkList &L)
{
    if (L) // 已初始化
    {
        LinkList p = L;
        while (L)
        {
            L = L->next;
            free(p);
            p = L;
        }
        return OK;
    }
    else // 未初始化
        return INFEASIBLE;
}

/* 清空链表元素 */
status ClearList(LinkList &L)
{
    if (L)
    {
        LinkList p = L->next;
        L->next = NULL;
        DestroyList(p); // 释放元素节点空间但不释放头节点空间
        return OK;
    }
    else
        return INFEASIBLE;
}

/* 判断链表是否为空 */
status ListEmpty(LinkList L)
{
    if (L) // 已初始化
    {
        if (L->next) // 头节后继不为空
            return FALSE;
        else // 头节点后继为空
            return TRUE;
    }
    else
        return INFEASIBLE;
}

/* 链表长度 */
int ListLength(LinkList L)
{
    if (L)
    {
        int i = 0;      // 元素个数
        while (L->next) // 未到表尾
        {
            i++;
            L = L->next; // 后继
        }
        return i; // 返回长度
    }
    else
        return INFEASIBLE;
}

/* 获取第i个元素 */
status GetElem(LinkList L, int i, ElemType &e)
{
    if (L)
    {
        int len = ListLength(L); // 长度
        if (i <= 0 || i > len)   // 查询的位置不合法
            return ERROR;
        else
        {
            while (i--)
                L = L->next;
            e = L->data;
            return OK;
        }
    }
    else
        return INFEASIBLE;
}

/* 获取元素位置 */
int LocateElem(LinkList L, ElemType e)
{
    if (L)
    {
        int i = 0;
        L = L->next; // 从首元开始
        while (L)    // 未到表尾
        {
            i++;
            if (L->data == e)
                return i; // 查询到
            L = L->next;
        }
        return ERROR; // 未查询到，元素不存在
    }
    else
        return INFEASIBLE;
}

/* 获取前驱元素 */
status PriorElem(LinkList L, ElemType e, ElemType &pre)
{
    if (L)
    {
        if (L->next == NULL) // 长度为零
            return ERROR;
        if (L->next->data == e) // 首元无前驱
            return ERROR;
        else
        {
            L = L->next; // 从首元开始
            while (L->next)
            {
                if (L->next->data == e)
                {
                    pre = L->data;
                    return OK;
                }
                L = L->next;
            }
            return ERROR;
        }
    }
    else
        return INFEASIBLE;
}

/* 获取后继元素 */
status NextElem(LinkList L, ElemType e, ElemType &next)
{
    if (L)
    {
        if (L->next == NULL) // 长度为零
            return ERROR;
        L = L->next;
        while (L) // 未到表尾
        {
            if (L->data == e)
            {
                if (L->next == NULL) // 尾元无后继
                    return ERROR;
                else
                {
                    next = L->next->data;
                    return OK;
                }
            }
            L = L->next;
        }
        return ERROR;
    }
    else
        return INFEASIBLE;
}

/* 指定位置前插入元素 */
status ListInsert(LinkList &L, int i, ElemType e)
{
    if (!L)
        return INFEASIBLE;
    int len = ListLength(L);
    if (i < 1 || i > len + 1) // 插入位置不合法
        return ERROR;
    LinkList ll = L; // L始终为头节点，不能改变
    while (--i)
        ll = ll->next; // 找到插入位置
    LinkList p = ll->next;
    LinkList temp = (LinkList)malloc(sizeof(LNode)); // 生成新节点
    temp->data = e;
    ll->next = temp;
    temp->next = p;
    return OK;
}

/* 删除指定位置元素 */
status ListDelete(LinkList &L, int i, ElemType &e)
{
    if (!L)
        return INFEASIBLE;
    int len = ListLength(L);
    if (i < 1 || i > len) // 删除位置不合法
        return ERROR;
    LinkList ll = L;
    while (--i)
        L = L->next; // 找到删除位置
    LinkList p = L->next;
    e = p->data;
    L->next = L->next->next;
    free(p); // 释放该节点空间
    L = ll;
    return OK;
}

/* 用于遍历时输出 */
void visit(ElemType item)
{
    // 输出当前遍历到的元素
    printf("%d ", item);
}

/* 显示当前链表所有元素 */
status ListTraverse(LinkList L, void (*visit)(ElemType))
{
    // L不是引用参数，可以直接使用，不会改变主函数中的L
    if (!L)
        return INFEASIBLE;
    while (L->next) // 未到表尾
    {
        L = L->next;
        visit(L->data); // 逐个遍历
    }
    return OK;
}

/* 逆置当前链表 */
void ListReverse(LinkList &L)
{
    // 1.递归算法
    // LinkList p = L->next;
    // if (L->next == NULL || p->next == NULL)
    //     return;
    // L->next = p->next;
    // ListReverse(L);
    // p->next->next = p;
    // p->next = NULL;

    // 2.用p指向首元节点，再将头节点的指针域赋空，用p遍链表，采用头插法插入
    LinkList p = L->next; // p指向首元节点
    L->next = NULL;       // 头节点指针域赋空
    LinkList q;
    while (p)
    {
        q = p->next; // 保存当前p的后继
        p->next = L->next;
        L->next = p;
        p = q;
    }
}

/* 删除倒数第n个元素 */
void RemoveNthFromEnd(LinkList &L, int n)
{
    int len = ListLength(L); // 长度
    ElemType e;
    if (ListDelete(L, len - n + 1, e) == OK) // 删除成功
        printf("Successfully deleted.");
    else if (ListDelete(L, len - n + 1, e) == ERROR) // 删除失败
        printf("The position is illegal.\n");
}

/* 给当前链表排序 */
void SortList(LinkList &L)
{
    // 交换数据域的冒泡排序
    int len = ListLength(L);
    LinkList p = L->next;
    for (int i = 1; i < len; i++) // 共进行len-1次，每次把最大数移到末尾
    {
        p = L->next;                      // 每一轮应该从第一个元素开始比较
        for (int j = 0; j < len - i; j++) // 每一轮只比较到第len-i-1和第len-i个元素
        {
            int temp;
            if (p->data > p->next->data)
            {
                // 交换数据域
                temp = p->data;
                p->data = p->next->data;
                p->next->data = temp;
            }
            p = p->next;
        }
    }
}

/* 保存到文件 */
void SaveData(LISTS Lists)
{
    FILE *fp = fopen(FileName, "w"); // 覆盖写入
    if (fp == NULL)                  // 尝试打开，如果文件不存在，则创建文件
        fp = fopen(FileName, "wb");
    LinkList p;
    for (int i = 0; i < Lists.length; i++)
    {
        // 按一定格式保存到文件
        if (Lists.elem[i].L != NULL) // 初始化的才会被保存
        {
            fprintf(fp, "name:%s length:%d\n", Lists.elem[i].name, ListLength(Lists.elem[i].L));
            p = Lists.elem[i].L->next;
            while (p)
            {
                fprintf(fp, "%d\n", p->data);
                p = p->next;
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
    return;
}

/* 从文件中加载 */
void LoadData(LISTS &Lists)
{
    FILE *fp = fopen(FileName, "r"); // 尝试打开文件
    if (fp == NULL)                  // 如果文件不存在
    {
        printf("File doesn't exist\n");
        return;
    }
    char current_list_name[MAX_NAME_LENGTH];
    int current_list_length;
    ElemType current_elem;
    Lists.length = 0; // 从0开始，原链表集合被覆盖
    while (Lists.length < MAX_LIST_NUM && fscanf(fp, "name:%s length:%d\n", current_list_name, &current_list_length) != EOF)
    {
        // 打印log
        printf("Reading a list with the name %s.\n", current_list_name);
        free(Lists.elem[Lists.length].L);     // 释放原有空间
        Lists.elem[Lists.length].L = NULL;    // 赋空
        InitList(Lists.elem[Lists.length].L); // 重新分配空间
        strcpy(Lists.elem[Lists.length].name, current_list_name);
        for (int i = 0; i < current_list_length; i++)
        {
            fscanf(fp, "%d\n", &current_elem);
            printf("element %d is being read.\n", current_elem);
            ListInsert(Lists.elem[Lists.length].L, i + 1, current_elem); // 插入元素
        }
        printf("\n");
        fscanf(fp, "\n");
        Lists.length++; // 长度加一
    }
    fclose(fp);
    return;
}
