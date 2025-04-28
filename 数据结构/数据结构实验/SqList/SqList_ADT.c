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
#define LIST_INIT_SIZE 100  // 线性表初始容量
#define LISTINCREMENT 10    // 线性表扩容容量
#define MAX_LIST_NUM 10     // 线性表数量最大值
#define MAX_NAME_LENGTH 30  // 每个线性表名称长度最大值
#define FileName "data.txt" // 文件名
// char FileName[MAX_NAME_LENGTH];

/* 定义数据类型 */
typedef int status;
typedef int ElemType; // 数据元素类型定义

/* 线性表（顺序结构）的定义 */
typedef struct
{
    ElemType *elem; // 存储空间基址
    int length;     // 当前长度
    int listsize;   // 当前分配的存储容量
} SqList;

/* 线性表的集合类型定义 */
typedef struct
{
    struct
    {
        char name[MAX_NAME_LENGTH]; // 线性表名称
        SqList *L;                  // 指向线性表的指针
    } elem[MAX_LIST_NUM];
    int length; // 当前线性表数量
} LISTS;

/* 全局变量 */
LISTS Lists;     // 线性表集合Lists
int current = 0; // 当前线性表在Lists中的位置

/* 函数声明 */
void printMenu();                                         // 打印菜单
void clearAllList(LISTS Lists);                           // Lists初始化
status checkList(SqList *L);                              // 检查线性表合法性
void visit(ElemType item);                                // 用于遍历时输出
status InitList(SqList *L);                               // 初始化线性表
status DestroyList(LISTS *Lists, int current);            // 销毁线性表
status ClearList(SqList *L);                              // 清空线性表
status ListEmpty(SqList L);                               // 判断线性表是否为空
int ListLength(SqList L);                                 // 获取线性表长度
status GetElem(SqList L, int i, ElemType *e);             // 获取线性表指定位置元素
int LocateElem(SqList L, ElemType e);                     // 定位元素在线性表中的位置
status PriorElem(SqList L, ElemType e, ElemType *pre);    // 获取指定元素的前驱
status NextElem(SqList L, ElemType e, ElemType *next);    // 获取指定元素的后继
status ListInsert(SqList *L, int i, ElemType e);          // 在指定位置插入元素
status ListDelete(SqList *L, int i, ElemType *e);         // 删除指定位置元素
status ListTraverse(SqList L, void (*visit)(ElemType));   // 遍历线性表
status SortCurrent(SqList *L);                            // 对当前线性表排序
ElemType MaxSubArray(SqList L);                           // 获取当前线性表的最大子数组和
int SubArrayNum(SqList L, ElemType k);                    // 获取当前线性表中和为k的连续子数组个数
void ShowAllList(LISTS Lists);                            // 显示所有线性表
status ChangeList(char ListName[], int *current);         // 切换当前线性表
status RemoveList(LISTS *Lists, char ListName[], int *p); // 移除指定线性表
status SaveData(LISTS Lists);                             // 将线性表数据保存到文件
status LoadData(LISTS *LL);                               // 从文件加载线性表数据
int search(LISTS Lists, char name[]);                     // 查找线性表位置

/* 打印菜单 */
void printMenu()
{
    printf("|---------Menu for Linear Table On Sequence Structure---------|\n");
    printf("|                                                             |\n");
    printf("|      1.  Create a List          2.  Destroy Current List    |\n");
    printf("|      3.  Clear Current List     4.  Empty or Not            |\n");
    printf("|      5.  Show List Length       6.  Get Element             |\n");
    printf("|      7.  Locate Element         8.  Get Prior Element       |\n");
    printf("|      9.  Get Next Element       10. Insert Element          |\n");
    printf("|      11. Delete Element         12. Show Current List       |\n");
    printf("|      13. Sort Current List      14. Max Sub Array           |\n");
    printf("|      15. Sub Array Num          16. Show All Lists          |\n");
    printf("|      17. Change Current List    18. Remove a List           |\n");
    printf("|      19. Save All Data          20. Load All Data           |\n");
    printf("|      21. Search a List          0.  EXIT                    |\n");
    printf("|                                                             |\n");
    printf("|-------------------------------------------------------------|\n\n");
    // printf("|-------Please Choose Your Operation from Options above-------|\n");
    // printf("|-------------------------------------------------------------|\n\n");
}

/* Lists初始化，将所有线性表指针置为空 */
void clearAllList(LISTS Lists)
{
    // 遍历所有线性表指针，将其置为空
    for (int i = 0; i < MAX_LIST_NUM; i++)
        Lists.elem[i].L = NULL;
}

/* 检查当前线性表是否合法 */
status checkList(SqList *L)
{
    // 检查线性表指针是否为空，为空则输出提示信息
    if (!L)
    {
        printf("The linear table does not exist.\n");
        printf("You can enter 1 to create a list or 17 to choose a list.\n");
        return FALSE;
    }
    // 检查线性表是否已经初始化，未初始化则输出提示信息
    else if (!L->elem)
    {
        printf("You need to initialize first.\n");
        return FALSE;
    }
    // 线性表合法，返回TRUE
    else
        return TRUE;
}

/* 用于遍历时输出 */
void visit(ElemType item)
{
    // 输出当前遍历到的元素
    printf("%d ", item);
}

/* 主函数 */
int main()
{
    // system("color 38");
    printMenu();         // 打印菜单
    clearAllList(Lists); // 初始化Lists
    SqList *L = NULL;    // 当前线性表指针
    Lists.length = 0;    // 当前线性表数量初始化为0
    int op = 1;          // 操作选项
    while (op)
    {
        // 提示用户选择操作
        printf("\n|-------------------------------------------------------------|\n");
        printf("|-------Please Choose Your Operation from Options above-------|\n");
        printf("|-------------------------------------------------------------|\n\n");
        scanf("%d", &op); // 读取用户输入
        system("cls");    // 清屏
        printMenu();
        switch (op)
        {
        case 1: // 创建线性表
            printf("|-------------------------------------------------------------|\n");
            printf("|---------You can create a total of %d linear tables.---------|\n", MAX_LIST_NUM); // 可创建线性表总数
            // 当前创建线性表的数量
            if (Lists.length > 1)
                printf("|--------Currently, %d linear tables have been created.--------|\n", Lists.length);
            else
                printf("|---------Currently, %d linear table has been created.---------|\n", Lists.length);
            printf("|-------------------------------------------------------------|\n\n");
            if (Lists.length < MAX_LIST_NUM) // 容量未满
            {
                printf("|-------------------------------------------------------------|\n");
                printf("|--When you create a linear table, it is selected by default--|\n");
                printf("|-------------------------------------------------------------|\n");
                printf("\nPlease enter the name of the linear table you want to add : \n");
                char s[MAX_NAME_LENGTH];
                scanf("%s", s); // 新增线性表名称
                int flag = 1;   // 新增的线性表是否存在
                for (int i = 0; i < Lists.length; i++)
                {
                    // 在已有线性表中查找
                    if (strcmp(Lists.elem[i].name, s) == 0)
                        flag = 0;
                }
                if (flag) // 新增线性表不存在
                {
                    Lists.elem[Lists.length].L = (SqList *)malloc(sizeof(SqList));
                    strcpy(Lists.elem[Lists.length].name, s);
                    L = Lists.elem[Lists.length].L; // 当前线性表指针指向新创建的线性表
                    current = Lists.length;         // 更新当前位置
                    L->elem = NULL;                 // 初始化为空
                    Lists.length++;                 // 线性表数量加一
                    // 初始化
                    if (InitList(L) == OK)
                        printf("The linear table (name: %s) is created!\n", s);
                    else
                        printf("Failed to create a linear table!\n");
                }
                else // 新增线性表存在
                    printf("The linear table already exists.\n");
                getchar();
                break;
            }
            else
            {
                printf("Capacity is full!\n"); // 容量已满
                getchar();
                break;
            }
        case 2: // 销毁当前线性表
            if (checkList(L))
            {
                if (DestroyList(&Lists, current) == OK) // 销毁成功
                {
                    L = NULL;
                    printf("The linear table was successfully destroyed.\n");
                }
            }
            getchar();
            break;
        case 3: // 清空当前线性表
            if (checkList(L))
            {
                if (ClearList(L) == OK) // 清空成功
                    printf("The linear table was successfully cleared.\n");
            }
            getchar();
            break;
        case 4: // 判断当前线性表是否为空
            if (checkList(L))
            {
                if (ListEmpty(*L) == TRUE) // 为空
                    printf("The linear table is empty.\n");
                else if (ListEmpty(*L) == FALSE) // 不为空
                    printf("The linear table is not empty.\n");
            }
            getchar();
            break;
        case 5: // 获取当前线性表长度
            if (checkList(L))
            {
                if (ListLength(*L) != INFEASIBLE) // 返回长度
                    printf("The length of the linear table is:%d\n", ListLength(*L));
            }
            getchar();
            break;
        case 6: // 获取指定位置元素
            if (checkList(L))
            {
                printf("Please enter the position (between 1 to %d) you want to query:\n", ListLength(*L));
                int queryPosition; // 查询的位置
                scanf("%d", &queryPosition);
                ElemType queryResult;                               // 查询的结果
                if (GetElem(*L, queryPosition, &queryResult) == OK) // 位置合法
                    printf("The element is %d.\n", queryResult);
                else if (GetElem(*L, queryPosition, &queryResult) == ERROR) // 位置不合法
                    printf("The position is illegal.\n");
            }
            getchar();
            break;
        case 7: // 定位元素在线性表中的位置
            if (checkList(L))
            {
                printf("Please enter the element you want to locate.\n");
                ElemType queryElem_locate; // 查询的元素
                scanf("%d", &queryElem_locate);
                if (LocateElem(*L, queryElem_locate) != ERROR) // 存在
                    printf("The position of %d is %d.\n", queryElem_locate, LocateElem(*L, queryElem_locate));
                else
                    printf("The element does not exist.\n"); // 不存在
            }
            getchar();
            break;
        case 8: // 获取指定元素的前驱
            if (checkList(L))
            {
                printf("Please enter the element you want to query:\n");
                ElemType queryElem_prior, pre; // 查询的元素及前驱元素
                scanf("%d", &queryElem_prior);
                if (PriorElem(*L, queryElem_prior, &pre) == OK) // 位置合法且元素存在
                    printf("The prior element of %d is %d.\n", queryElem_prior, pre);
                else if (PriorElem(*L, queryElem_prior, &pre) == ERROR) // 位置不合法或元素不存在
                    printf("failed to find.\n");
            }
            getchar();
            break;
        case 9: // 获取指定元素的后继
            if (checkList(L))
            {
                printf("Please enter the element you want to query:\n");
                ElemType queryElem_next, next; // 查询的元素及后继元素
                scanf("%d", &queryElem_next);
                if (NextElem(*L, queryElem_next, &next) == OK) // 位置合法且元素存在
                    printf("The next element of %d is %d.\n", queryElem_next, next);
                else if (NextElem(*L, queryElem_next, &next) == ERROR) // 位置不合法或元素不存在
                    printf("failed to find.\n");
            }
            getchar();
            break;
        case 10: // 在指定位置插入元素
            if (checkList(L))
            {
                printf("Position: (between 1 to %d)\n", ListLength(*L) + 1); // 可供选择的位置
                printf("Please enter the position and the element you want to insert:(spaced by space)\n");
                int insertPosition;  // 插入的位置
                ElemType insertElem; // 插入的元素
                scanf("%d %d", &insertPosition, &insertElem);
                if (ListInsert(L, insertPosition, insertElem) == OK) // 插入成功
                    printf("Successfully inserted.\n");
                else if (ListInsert(L, insertPosition, insertElem) == ERROR) // 插入失败
                    printf("The position is illegal.\n");
            }
            getchar();
            break;
        case 11: // 删除指定位置元素
            if (checkList(L))
            {
                printf("Position: (between 1 to %d)\n", ListLength(*L)); // 可供选择的位置
                printf("Please enter the position you want to delete:\n");
                int deletePosition;  // 删除的位置
                ElemType deleteElem; // 删除的元素
                scanf("%d", &deletePosition);
                if (ListDelete(L, deletePosition, &deleteElem) == OK) // 删除成功
                    printf("Delete %d in position %d.\n", deleteElem, deletePosition);
                else if (ListDelete(L, deletePosition, &deleteElem) == ERROR) // 删除失败
                    printf("The position is illegal.\n");
            }
            getchar();
            break;
        case 12: // 显示当前线性表
            if (checkList(L))
            {
                if (ListTraverse(*L, visit) == OK) // 成功遍历
                    printf("Successfully traveled all elements.\n");
            }
            getchar();
            break;
        case 13: // 对当前线性表进行排序
            if (checkList(L))
            {
                if (SortCurrent(L) == OK) // 成功排序
                    printf("Successfully sorted.\n");
            }
            getchar();
            break;
        case 14: // 获取当前线性表的最大子数组和
            if (checkList(L))
            {
                if (L->length)
                    printf("Max Sub=%d", MaxSubArray(*L)); // 最大和
                else
                    printf("List length = 0, failed to find.\n"); // 表中无元素
            }
            getchar();
            break;
        case 15: // 获取当前线性表中和为k的连续子数组个数
            if (checkList(L))
            {
                if (L->length)
                {
                    printf("Please enter the sum of the continuous subarrays you want to query:\n");
                    int k;
                    scanf("%d", &k);
                    int num = SubArrayNum(*L, k);
                    if (num > 1) // 和为k的子数组个数
                        printf("There are %d continuous subarrays with an sum of %d.\n", num, k);
                    else
                        printf("There is %d continuous subarray with an sum of %d.\n", num, k);
                }
                else
                    printf("List length = 0, failed to find.\n"); // 表中无元素
            }
            getchar();
            break;
        case 16:                   // 显示所有线性表
            if (Lists.length == 0) // 表中无元素
                printf("There are no linear tables.\n");
            else
                ShowAllList(Lists);
            getchar();
            break;
        case 17: // 切换当前线性表
            printf("Please enter the name you want to change to:\n");
            char temp_change[MAX_NAME_LENGTH]; // 要切换的表的名称
            scanf("%s", temp_change);
            if (ChangeList(temp_change, &current) == OK) // 切换成功
            {
                L = Lists.elem[current].L; // L指向切换后线性表
                printf("Successfully changed.\n");
            }
            else // 表不存在
                printf("There is no linear table named %s.\n", temp_change);
            getchar();
            break;
        case 18: // 删除指定线性表
            printf("Please enter the name you want to remove:\n");
            char temp_remove[MAX_NAME_LENGTH]; // 要删除的表的名称
            scanf("%s", temp_remove);
            int p;
            if (RemoveList(&Lists, temp_remove, &p) == OK) // 该表存在
            {
                printf("Successfully removed.\n");
                if (p == current) // 如果删除的是当前表
                    L = NULL;
                else if (p < current)
                    current -= 1; // 位置前移
            }
            else // 该表不存在
                printf("There is no linear table named %s.\n", temp_remove);
            getchar();
            break;
        case 19: // 将线性表数据保存到文件
            SaveData(Lists);
            printf("Successfully Saved.\n"); // 成功保存
            getchar();
            break;
        case 20: // 从文件加载线性表数据
            printf("Are you sure you want to read from the file?\n");
            printf("The data that is not currently saved will be gone.\n");
            printf("confirm:1  cancel:0\n");
            // 从文件中读取会覆盖当前Lists
            int choice;
            scanf("%d", &choice);
            if (choice)
            {
                if (LoadData(&Lists) == OK) // 参数回传，更新当前Lists
                // LoadData();
                {
                    L = NULL;
                    printf("Successfully Loaded.\n");
                    printf("Now you can enter 16 to query all linear tables in the file.");
                }
            }
            getchar();
            break;
        case 21: // 查找线性表位置
            printf("Please enter the name of the linear table you want to query:\n");
            char search_name[MAX_NAME_LENGTH];
            scanf("%s", search_name);
            if (search(Lists, search_name) != -1)
                printf("The location of the linear table is %d.\n", search(Lists, search_name));
            else
                printf("There is no linear table named %s.\n", search_name);
            getchar();
            break;
        case 0:
            break;
        default:
            printf("The feature number is incorrect.\n"); // 功能选项错误
        } // end of switch
    } // end of while
    printf("Welcome to use this system next time!\n");
    return 0;
}

/* 初始化线性表 */
status InitList(SqList *L)
{
    if (L == NULL) // 表不存在
        return ERROR;
    if (L->elem == NULL) // 未分配空间
    {
        L->elem = (int *)malloc(sizeof(int) * LIST_INIT_SIZE); // 分配空间
        L->listsize = LIST_INIT_SIZE;                          // 初始容量
        L->length = 0;                                         // 初始长度为0
        return OK;
    }
    else // 已初始化
        return INFEASIBLE;
}

/* 销毁线性表 */
status DestroyList(LISTS *Lists, int current)
{
    free(Lists->elem[current].L); // 销毁空间
    for (int i = current; i < Lists->length; i++)
    {
        Lists->elem[i] = Lists->elem[i + 1];
    }
    Lists->length--; // 数量减一
    return OK;
}

/* 清空线性表 */
status ClearList(SqList *L)
{
    if (L->elem == NULL) // 未初始化
        return INFEASIBLE;
    else
    {
        L->length = 0; // 长度清零
        return OK;
    }
}

/* 判断线性表是否为空 */
status ListEmpty(SqList L)
{
    if (L.elem == NULL) // 未初始化
        return INFEASIBLE;
    else
    {
        if (L.length == 0)
            return TRUE; // 空
        else
            return FALSE; // 非空
    }
}

/* 获取线性表长度 */
int ListLength(SqList L)
{
    if (L.elem == NULL) // 未初始化
        return INFEASIBLE;
    else
        return L.length; // 返回长度
}

/* 获取线性表指定位置元素 */
status GetElem(SqList L, int i, ElemType *e)
{
    if (L.elem == NULL) // 未初始化
        return INFEASIBLE;
    else if (i <= 0 || i > L.length)
        return ERROR; // 长度不合法
    else
    {
        *e = L.elem[i - 1]; // 参数回传
        return OK;
    }
}

/* 定位元素在线性表中的位置 */
int LocateElem(SqList L, ElemType e)
{
    if (L.elem == NULL) // 未初始化
        return INFEASIBLE;
    else
    {
        int i = 0;
        for (i; i < L.length; i++)
        {
            if (L.elem[i] == e) // e存在
                return i + 1;   // 返回位置
        }
        if (i >= L.length) // e不存在
            return ERROR;
    }
}

/* 获取指定元素的前驱 */
status PriorElem(SqList L, ElemType e, ElemType *pre)
{
    if (L.elem == NULL) // 未初始化
        return INFEASIBLE;
    else
    {
        int i = 0;
        for (i; i < L.length; i++)
        {
            if (L.elem[i] == e) // e存在
            {
                if (i == 0) // 位置不合法
                    return ERROR;
                else
                {
                    *pre = L.elem[i - 1]; // 参数回传
                    return OK;
                }
            }
        }
        if (i >= L.length) // e不存在
            return ERROR;
    }
}

/* 获取指定元素的后继 */
status NextElem(SqList L, ElemType e, ElemType *next)
{
    if (L.elem == NULL) // 未初始化
        return INFEASIBLE;
    else
    {
        int i = 0;
        for (i; i < L.length; i++)
        {
            if (L.elem[i] == e) // e存在
            {
                if (i == L.length - 1) // 位置不合法
                    return ERROR;
                else
                {
                    *next = L.elem[i + 1]; // 参数回传
                    return OK;
                }
            }
        }
        if (i >= L.length) // e不存在
            return ERROR;
    }
}

/* 在指定位置插入元素 */
status ListInsert(SqList *L, int i, ElemType e)
{
    if (L->elem == NULL) // 未初始化
        return INFEASIBLE;
    if (i <= 0 || i > L->length + 1) // 位置不合法
        return ERROR;
    else
    {
        if (L->length >= L->listsize) // 容量不够
        {
            // 重新分配空间
            ElemType *newbase;
            newbase = (ElemType *)realloc(L->elem, sizeof(ElemType) * L->listsize + LISTINCREMENT);
            if (newbase)
            {
                L->elem = newbase;
                L->listsize += LISTINCREMENT;
            }
        }
    }
    for (int j = L->length - 1; j >= i - 1; j--) // 后移
        L->elem[j + 1] = L->elem[j];
    L->elem[i - 1] = e; // 插入
    L->length++;        // 长度加一
    return OK;
}

/* 删除指定位置元素 */
status ListDelete(SqList *L, int i, ElemType *e)
{
    if (L->elem == NULL) // 未初始化
        return INFEASIBLE;
    if (i <= 0 || i > L->length) // 位置不合法
        return ERROR;
    *e = L->elem[i - 1]; // 参数回传
    for (int j = i - 1; j < L->length - 1; j++)
        L->elem[j] = L->elem[j + 1]; // 前移
    L->length--;                     // 长度减一
    return OK;
}

/* 遍历线性表 */
status ListTraverse(SqList L, void (*visit)(ElemType))
{
    if (L.elem == NULL) // 未初始化
        return INFEASIBLE;
    if (L.length)
    {
        int literate_time = 0; // 未初始化
        for (; literate_time < L.length; literate_time++)
        { // 对每一个元素执行visit函数，此处visit函数的作用是打印元素
            visit(L.elem[literate_time]);
        }
        printf("\n");
        return OK;
    }
    else // 长度为零
    {
        printf("List length = 0, failed to travel.\n");
        return ERROR;
    }
}

/* 对当前线性表进行排序 */
status SortCurrent(SqList *L)
{
    if (L->length) // 存在元素
    {
        // 冒泡排序
        for (int i = L->length - 1; i > 0; i--)
        {
            for (int j = 0; j < i; j++)
            {
                if (L->elem[j] > L->elem[j + 1])
                {
                    int temp = L->elem[j];
                    L->elem[j] = L->elem[j + 1];
                    L->elem[j + 1] = temp;
                }
            }
        }
        return OK;
    }
    else // 不存在元素
    {
        printf("List length = 0, failed to sort.\n");
        return ERROR;
    }
}

/* 获取当前线性表的最大子数组和 */
ElemType MaxSubArray(SqList L)
{
    if (L.length) // 存在元素
    {
        // 初始化最大和和当前子数组的和为数组的第一个元素
        int max_sum = L.elem[0];
        int current_sum = L.elem[0];
        // 从数组的第二个元素开始遍历
        for (int i = 1; i < L.length; i++)
        {
            // 更新当前子数组的和，使其为当前元素值与当前元素值与前一个子数组的和中的较大值
            current_sum = (current_sum + L.elem[i] > L.elem[i]) ? current_sum + L.elem[i] : L.elem[i];
            // 更新最大和，使其为当前子数组的和和最大和中的较大值
            max_sum = (current_sum > max_sum) ? current_sum : max_sum;
        }
        return max_sum;
    }
}

/* 获取当前线性表中和为k的连续子数组个数 */
/* -2 1 -3 4 -1 2 1 -5 4 */
int SubArrayNum(SqList L, ElemType k)
{
    int count = 0;
    // 遍历数组，计算以每个元素为起点的子数组的和
    for (int i = 0; i < L.length; i++)
    {
        int sum = 0;
        // 将当前元素与后续元素相加，直到到达数组末尾
        for (int j = i; j < L.length; j++)
        {
            sum += L.elem[j];
            // 如果子数组和等于 k，增加计数
            if (sum == k)
            {
                count++;
                // break;
            }
        }
    }
    return count;
}

/* 显示所有线性表 */
void ShowAllList(LISTS Lists)
{
    // 遍历Lists的每一个线性表
    for (int i = 0; i < Lists.length; i++)
    {
        printf("name:%s\n", Lists.elem[i].name);
        printf("elements:");
        for (int j = 0; j < Lists.elem[i].L->length; j++)
            visit(Lists.elem[i].L->elem[j]);
        printf("\n");
    }
}

/* 切换当前线性表 */
status ChangeList(char ListName[], int *current)
{
    for (int i = 0; i < Lists.length; i++)
    {
        if (strcmp(Lists.elem[i].name, ListName) == 0) // 存在
        {
            *current = i; // 更新当前位置
            return OK;
        }
    }
    return ERROR; // 不存在
}

/* 移除指定线性表 */
status RemoveList(LISTS *Lists, char ListName[], int *p)
{
    for (int i = 0; i < Lists->length; i++)
    {
        if (strcmp(Lists->elem[i].name, ListName) == 0) // 存在
        {
            free(Lists->elem[i].L); // 释放空间
            *p = i;                 // 回传删除的位置
            for (int j = i; j < Lists->length; j++)
            {
                Lists->elem[i] = Lists->elem[i + 1]; // 前移
            }
            Lists->length--; // 数量减一
            return OK;
        }
    }
    return ERROR; // 不存在
}

/* 将线性表数据保存到文件 */
status SaveData(LISTS Lists)
{
    // printf("Please enter the filename:\n");
    // scanf("%s",FileName);
    FILE *fp = fopen(FileName, "w"); // 覆盖写入
    if (fp == NULL)                  // 尝试打开，如果文件不存在，则创建文件
        fp = fopen(FileName, "wb");
    int literate_time = 0;
    for (; literate_time < Lists.length; literate_time++)
    {
        if (Lists.elem[literate_time].L)
        {
            // 按照一定格式将数据保存到文件中
            fprintf(fp, "name:%s length:%d\n", Lists.elem[literate_time].name, Lists.elem[literate_time].L->length);
            for (int i = 0; i < Lists.elem[literate_time].L->length; i++)
                fprintf(fp, "%d\n", Lists.elem[literate_time].L->elem[i]);
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
    return OK;
}

/* 从文件加载线性表数据 */
status LoadData(LISTS *LL)
// 还有一种读取方法为仅显示，但当前Lists并不会更新为文件中内容。
// Lists只是暂存的，如果没有Save就去Load，当前暂存的Lists会被文件内容覆盖。
{
    // printf("Please enter the filename:\n");
    // scanf("%s",FileName);
    FILE *fp = fopen(FileName, "r"); // 尝试打开文件
    if (fp == NULL)                  // 如果文件不存在
    {
        printf("File doesn't exist\n");
        return ERROR;
    }
    int literate_time = 0;                   // 当前位置
    char current_list_name[MAX_NAME_LENGTH]; // 当前读取的线性表的名字
    ElemType current_elem;                   // 当前读取的元素
    int list_length;                         // 当前线性表长度
    LL->length = 0;                          // 初始数量为零
    // 不断读取直到文件尾，即EOF
    while (literate_time < MAX_LIST_NUM && fscanf(fp, "name:%s length:%d\n", current_list_name, &list_length) != EOF)
    {
        // 打印log
        printf("Reading a linear table with the name %s.\n", current_list_name);

        free(LL->elem[literate_time].L);                              // 释放原有空间
        LL->elem[literate_time].L = (SqList *)malloc(sizeof(SqList)); // 重新分配空间
        strcpy(LL->elem[literate_time].name, current_list_name);
        LL->elem[literate_time].L->length = list_length;
        LL->elem[literate_time].L->listsize = LIST_INIT_SIZE;
        LL->elem[literate_time].L->elem = (ElemType *)malloc(sizeof(ElemType) * LIST_INIT_SIZE);
        for (int i = 0; i < list_length; i++)
        {
            fscanf(fp, "%d\n", &current_elem);
            printf("element %d is being read.\n", current_elem);
            (LL->elem[literate_time].L->elem)[i] = current_elem; // 读取元素
        }
        printf("\n");
        literate_time++; // 位置后移
        LL->length++;    // 数量加一
        fscanf(fp, "\n");
    }
    return OK;
}

/* 查找线性表位置 */
int search(LISTS Lists, char name[])
{
    for (int i = 0; i < Lists.length; i++)
    {
        if (strcmp(Lists.elem[i].name, name) == 0) // 存在
            return i + 1;
    }
    return -1; // 不存在
}

