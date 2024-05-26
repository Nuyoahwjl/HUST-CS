/*头文件声明*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*定义常量*/
#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASIBLE -1
#define MAX_OTHERS 20       // 其他信息的最大长度
#define FileName "data.txt" // 文件名
#define MAX_NAME_LENGTH 20  // 树的名字最大长度
#define MAX_TREE_NUM 10     // 树的最大数量
/*定义数据元素类型*/
typedef int status;
typedef int KeyType;
/*二叉树结点数据类型定义*/
typedef struct
{
    KeyType key;
    char others[MAX_OTHERS];
} TElemType;
/*二叉链表结点的定义*/
typedef struct BiTNode
{
    TElemType data;
    struct BiTNode *lchild, *rchild;
} BiTNode, *BiTree;
/*二叉树集合*/
typedef struct
{
    struct
    {
        BiTree T;
        char name[MAX_NAME_LENGTH];
    } elem[MAX_TREE_NUM];
    int length; // 树的数量
} TreeList;
TreeList TL; // 全局变量，树的集合

/*函数声明*/
void PrintMenu1();                                              // 打印主菜单
status CreateBiTree(BiTree &T, TElemType definition[], int &i); // 创建树
int SelectTree(TreeList TL, char name[]);                       // 选择树
status ClearBiTree(BiTree &T);                                  // 清空树
void main2(BiTree &T, int loc);                                 // 子页面
void PrintMenu2();                                              // 打印子菜单
status BiTreeEmpty(BiTree T);                                   // 判断树是否为空
int BiTreeDepth(BiTree T);                                      // 求树的深度
BiTree LocateNode(BiTree T, KeyType e);                         // 定位结点
status Assign(BiTree &T, KeyType e, TElemType value);           // 赋值
BiTree GetSibling(BiTree T, KeyType e);                         // 获得兄弟结点
void visit(BiTree T);                                           // 访问结点
status InsertNode(BiTree &T, KeyType e, int LR, TElemType c);   // 插入结点
status DeleteNode(BiTree &T, KeyType e);                        // 删除结点
BiTree GetParent(BiTree T, KeyType e);                          // 获得父结点
status PreOrderTraverse(BiTree T, void (*visit)(BiTree));       // 先序遍历
status InOrderTraverse(BiTree T, void (*visit)(BiTree));        // 中序遍历
status PostOrderTraverse(BiTree T, void (*visit)(BiTree));      // 后序遍历
status LevelOrderTraverse(BiTree T, void (*visit)(BiTree));     // 层序遍历
status SaveBiTree(TreeList TL);                                 // 保存树
status LoadBiTree(TreeList &TL);                                // 读取树
status InvertTree(BiTree &T);                                   // 翻转树
int Save_Len(BiTree T);                                         // 保存树的长度
int MaxPathSum(BiTree T);                                       // 求树的最大路径和
BiTree LowestCommonAncestor(BiTree T, KeyType e1, KeyType e2);  // 求树的最近公共祖先

/*主函数*/
int main()
{
    PrintMenu1();  // 打印主菜单
    TL.length = 0; // 初始化树的数量
    int op = 1;
    while (op)
    {
        // 提示用户选择操作
        printf("\n|-------------------------------------------------------|\n"
               "|----Please Choose Your Operation from Options above----|\n"
               "|-------------------------------------------------------|\n\n");
        scanf("%d", &op); // 输入功能选项
        system("cls");    // 清屏
        PrintMenu1();     // 打印主菜单
        switch (op)
        {
        case 1: // 创建树
        {
            printf("|-------------------------------------------------------|\n"
                   "|----------You can create a total of %d trees.----------|\n",
                   MAX_TREE_NUM); // 提示用户可以创建的树的数量
            if (TL.length > 1)    // 提示用户当前已经创建的树的数量
                printf("|---------Currently, %d trees have been created.---------|\n", TL.length);
            else
                printf("|----------Currently, %d tree has been created.----------|\n", TL.length);
            printf("|-------------------------------------------------------|\n\n");
            // 提示用户输入树的名字
            printf("Please input the name of the tree you want to create:\n");
            scanf("%s", TL.elem[TL.length].name);
            // 检查是否有相同名字的树
            if (SelectTree(TL, TL.elem[TL.length].name) != -1)
            {
                printf("ERROR:There is a tree with the same name!\n");
                break;
            }
            // 提示用户输入树的关键字和其他信息
            printf("Please input the key and others of the tree, end with -1(PreOrder):\n");
            TElemType definition[100];
            int i = 0;
            // 输入树的关键字和其他信息
            while (1)
            {
                scanf("%d", &definition[i].key);
                if (definition[i].key == -1)
                    break;
                scanf("%s", definition[i].others);
                i++;
            }
            definition[i].key = -1; // 结束标志
            int a = 0;
            if (CreateBiTree(TL.elem[TL.length].T, definition, a) == ERROR) // 创建树
            {
                printf("ERROR:There are same keys in the tree!\n");
                break;
            }
            printf("The tree has been created successfully!\n");
            TL.length++; // 树的数量加1
            break;
        }
        case 2: // 删除树
        {
            printf("Please input the name of the tree you want to delete:\n");
            char delete_name[MAX_NAME_LENGTH]; // 删除树的名字
            scanf("%s", delete_name);
            if (SelectTree(TL, delete_name) == -1) // 未找到树
            {
                printf("ERROR:There is no such tree!\n");
                break;
            }
            else
            {
                ClearBiTree(TL.elem[SelectTree(TL, delete_name)].T); // 销毁树
                for (int i = SelectTree(TL, delete_name); i < TL.length - 1; i++)
                {
                    TL.elem[i] = TL.elem[i + 1]; // 前移
                }
                TL.length--;
                printf("The tree has been deleted successfully!\n");
            }
            break;
        }
        case 3: // 选择树
        {
            printf("Please input the name of the tree you want to select:\n");
            char select_name[MAX_NAME_LENGTH];
            scanf("%s", select_name);
            int loc = SelectTree(TL, select_name); // 选择树的位置
            if (loc == -1)                         // 未找到树
            {
                printf("ERROR:There is no such tree!\n");
                break;
            }
            main2(TL.elem[loc].T, loc); // 调用子页面
            PrintMenu1();               // 从子页面跳回时再次打印主菜单
            break;
        }
        case 4: // 保存数据
        {
            if (TL.length == 0)
            {
                printf("There is no tree to save.\n"); // 没有树可保存
                break;
            }
            SaveBiTree(TL);
            printf("The data has been saved successfully!\n");
            break;
        }
        case 5: // 读取数据
        {
            if (LoadBiTree(TL) == ERROR)
            {
                printf("ERROR:There is no such file!\n"); // 文件不存在
                break;
            }
            printf("The data has been loaded successfully!\n"); // 数据已经加载成功
            break;
        }
        case 6: // 显示所有树
        {
            if (TL.length == 0) // 没有树可显示
            {
                printf("There is no tree to show.\n");
                break;
            }
            for (int i = 0; i < TL.length; i++) // 显示所有树的名字
            {
                printf("%s\n", TL.elem[i].name);
            }
            break;
        }
        case 7: // 查询树的位置
        {
            char search_name[MAX_NAME_LENGTH];
            printf("Please enter the name of the tree you want to query:\n");
            scanf("%s", search_name);
            if (SelectTree(TL, search_name) != -1) // 找到树
                printf("The location of the tree is %d.\n", SelectTree(TL, search_name) + 1);
            else
                printf("There is no tree named %s.\n", search_name);
            break;
        }
        case 0: // 退出
            break;
        default:
            printf("The feature number is incorrect.\n"); // 功能选项错误
        }
    }
    printf("Welcome to use this system next time!\n"); // 欢迎下次使用
    return 0;
}

/*子页面*/
void main2(BiTree &T, int loc)
{
    system("cls"); // 清屏
    PrintMenu2();  // 打印子菜单
    int op = 1;
    while (op)
    {
        // 提示用户选择操作
        printf("\n|-------------------------------------------------------|\n"
               "|----Please Choose Your Operation from Options above----|\n"
               "|-------------------------------------------------------|\n\n");
        scanf("%d", &op); // 输入功能选项
        system("cls");    // 清屏
        PrintMenu2();     // 打印子菜单
        switch (op)
        {
        case 1: // 初始化树
        {
            if (BiTreeEmpty(T))
            {
                printf("Please input the key and others of the tree, end with -1(PreOrder):\n");
                TElemType definition[100];
                int i = 0;
                // 输入树的关键字和其他信息
                while (1)
                {
                    scanf("%d", &definition[i].key);
                    if (definition[i].key == -1)
                        break;
                    scanf("%s", definition[i].others);
                    i++;
                }
                definition[i].key = -1; // 结束标志
                int a = 0;
                if (CreateBiTree(T, definition, a) == ERROR) // 创建树
                {
                    printf("ERROR:There are same keys in the tree!\n");
                    break;
                }
                printf("The tree has been initialized successfully!\n");
            }
            else // 树不为空
            {
                printf("The data already exist.\n");
                break;
            }
            break;
        }
        case 2: // 清空树
        {
            if (BiTreeEmpty(T)) // 树为空
            {
                break;
            }
            else // 树不为空
            {
                ClearBiTree(T); // 清空树
                printf("The tree has been cleared successfully!\n");
            }
            break;
        }
        case 3: // 销毁树
        {
            ClearBiTree(T);
            // 删除树
            for (int i = loc; i < TL.length - 1; i++)
            {
                TL.elem[i] = TL.elem[i + 1]; // 前移
            }
            TL.length--;
            // 树的数量减1
            system("cls"); // 清屏
            printf("\nThe tree has been destoryed successfully!\n\n");
            return;
            break;
        }
        case 4: // 判断树是否为空
        {
            if (BiTreeEmpty(T)) // 树为空
                ;
            else
                printf("The tree is not empty.\n"); // 树不为空
            break;
        }
        case 5: // 求树的深度
        {
            if (!BiTreeEmpty(T)) // 树不为空
                printf("The depth of the tree is %d.\n", BiTreeDepth(T));
            break;
        }
        case 6: // 定位结点
        {
            if (!BiTreeEmpty(T)) // 树不为空
            {
                KeyType e;
                printf("Please input the key you want to locate:\n");
                scanf("%d", &e);
                BiTree p = LocateNode(T, e);
                if (p == NULL) // 未找到结点
                    printf("The node is not found.\n");
                else // 找到结点
                    printf("The node is: %d,%s\n", p->data.key, p->data.others);
            }
            break;
        }
        case 7: // 赋值
        {
            if (!BiTreeEmpty(T)) // 树不为空
            {
                KeyType e;       // 关键字
                TElemType value; // 其他信息
                printf("Please input the key you want to assign:\n");
                scanf("%d", &e);
                printf("Please input the value you want to assign:\n");
                scanf("%d %s", &value.key, value.others);
                int s = Assign(T, e, value); // 赋值
                if (s == ERROR)              // 有相同的关键字
                    printf("ERROR:There is a same key in the tree!\n");
                else if (s == INFEASIBLE) // 未找到结点
                    printf("The node is not found.\n");
                else // 赋值成功
                    printf("The value has been assigned successfully!\n");
            }
            break;
        }
        case 8: // 获得兄弟结点
        {
            if (!BiTreeEmpty(T))
            {
                KeyType e;
                printf("Please input the key you want to get the sibling:\n");
                scanf("%d", &e);
                BiTree p = GetSibling(T, e); // 获得兄弟结点
                if (p == NULL)               // 未找到结点
                    printf("The node is not found.\n");
                else // 找到兄弟结点
                    printf("The sibling is: %d,%s\n", p->data.key, p->data.others);
            }
            break;
        }
        case 9: // 插入结点
        {
            if (!BiTreeEmpty(T))
            {
                KeyType e;
                int LR;
                TElemType c;
                printf("Please input the key you want to insert:\n");
                scanf("%d", &e); // 插入的位置
                printf("Please input the LR you want to insert(0 for left,1 for right,-1 for root):\n");
                scanf("%d", &LR); // 插入的方向
                printf("Please input the value you want to insert:\n");
                scanf("%d %s", &c.key, c.others); // 插入的值
                int s = InsertNode(T, e, LR, c);
                if (s == ERROR)
                    printf("ERROR:There is a same key in the tree!\n");
                else if (s == INFEASIBLE)
                    printf("The node is not found.\n");
                else
                    printf("The node has been inserted successfully!\n");
            }
            break;
        }
        case 10: // 删除结点
        {
            if (!BiTreeEmpty(T))
            {
                KeyType e;
                printf("Please input the key you want to delete:\n");
                scanf("%d", &e); // 删除的位置
                if (DeleteNode(T, e) == ERROR)
                    printf("The node is not found.\n");
                else
                    printf("The node has been deleted successfully!\n");
            }
            break;
        }
        case 11: // 先序遍历
        {
            if (!BiTreeEmpty(T))
            {
                printf("The PreOrderTraverse is:\n");
                PreOrderTraverse(T, visit);
                printf("\n");
            }
            break;
        }
        case 12: // 中序遍历
        {
            if (!BiTreeEmpty(T))
            {
                printf("The InOrderTraverse is:\n");
                InOrderTraverse(T, visit);
                printf("\n");
            }
            break;
        }
        case 13: // 后序遍历
        {
            if (!BiTreeEmpty(T))
            {
                printf("The PostOrderTraverse is:\n");
                PostOrderTraverse(T, visit);
                printf("\n");
            }
            break;
        }
        case 14: // 层序遍历
        {
            if (!BiTreeEmpty(T))
            {
                printf("The LevelOrderTraverse is:\n");
                LevelOrderTraverse(T, visit);
                printf("\n");
            }
            break;
        }
        case 15: // 求树的最大路径和
        {
            if (!BiTreeEmpty(T))
            {
                printf("The MaxPathSum is %d.\n", MaxPathSum(T));
            }
            break;
        }
        case 16: // 求树的最近公共祖先
        {
            if (!BiTreeEmpty(T))
            {
                KeyType e1, e2;
                printf("Please input the key of the first node:\n");
                scanf("%d", &e1);
                printf("Please input the key of the second node:\n");
                scanf("%d", &e2);
                BiTree p = LowestCommonAncestor(T, e1, e2);
                if (p == NULL)
                    printf("The nodes are not found.\n");
                else
                    printf("The LowestCommonAncestor is: %d,%s\n", p->data.key, p->data.others);
            }
            break;
        }
        case 17: // 翻转树
        {
            if (!BiTreeEmpty(T))
            {
                InvertTree(T);
                printf("The tree has been inverted successfully!\n");
            }
            break;
        }
        case 0:
            system("cls");
            return;
        default:
            printf("The feature number is incorrect.\n"); // 功能选项错误
        }
    }
}

/*主菜单*/
void PrintMenu1()
{
    printf("|================Menu for multiple Trees================|\n");
    printf("|-------------------------------------------------------|\n");
    printf("|              1.    Create a Tree                      |\n");
    printf("|              2.    Delete a Tree                      |\n");
    printf("|              3. Select a single Tree                  |\n");
    printf("|              4. Save All Data To File                 |\n");
    printf("|              5. Load All Data From File               |\n");
    printf("|              6.    Show All Trees                     |\n");
    printf("|              7.    Search a Tree                      |\n");
    printf("|              0.         EXIT                          |\n");
    printf("|=======================================================|\n\n");
}

/*子菜单*/
void PrintMenu2()
{
    printf("|================Menu for single Tree===================|\n");
    printf("|-------------------------------------------------------|\n");
    printf("|   1.  Init Current Tree     2.  Clear Current Tree    |\n");
    printf("|   3.  Destory Current Tree  4.  Empty or Not          |\n");
    printf("|   5.  Depth of the Tree     6.  Locate Node           |\n");
    printf("|   7.  Assign Value          8.  GetSibling            |\n");
    printf("|   9.  Insert a Node         10. Delete a Node         |\n");
    printf("|   11. PreOrderTraverse      12. InOrderTraverse       |\n");
    printf("|   13. PostOrderTraverse     14. LevelOrderTraverse    |\n");
    printf("|   15. MaxPathSum            16. LowestCommonAncestor  |\n");
    printf("|   17. InvertTree            0.  EXIT                  |\n");
    printf("|=======================================================|\n\n");
}

/*访问结点*/
void visit(BiTree T)
{
    printf("%d,%s ", T->data.key, T->data.others); // 访问结点
}

/*判空*/
status BiTreeEmpty(BiTree T)
{
    if (T == NULL) // 空树
    {
        printf("The tree is empty.\n");
        return TRUE;
    }
    else // 非空树
        return FALSE;
}

/*创建树*/
status CreateBiTree(BiTree &T, TElemType definition[], int &i)
{
    // 基于先序序列创建二叉树
    if (i == 0) // 第一次调用时检查是否有相同的关键字
    {
        for (int j = 0; definition[j].key != -1; j++)
        {
            for (int k = j + 1; definition[k].key != -1; k++)
            {
                if (definition[j].key == definition[k].key && definition[k].key != 0)
                    return ERROR; // 有相同的关键字
            }
        }
    }

    if (definition[i].key == -1)
        return OK; // 递归结束条件

    if (definition[i].key == 0) // 0表示空树
    {
        T = NULL;
        i++; // definition后移
        return OK;
    }
    else // 非空树
    {
        T = (BiTree)malloc(sizeof(BiTNode));
        T->data = definition[i];
        i++; // definition后移
        CreateBiTree(T->lchild, definition, i);
        CreateBiTree(T->rchild, definition, i);
    }
    return OK;
}

/*选择树*/
int SelectTree(TreeList TL, char name[])
{
    for (int i = 0; i < TL.length; i++)
    {
        if (strcmp(TL.elem[i].name, name) == 0)
            return i; // 找到树
    }
    return -1; // 未找到树
}

/*清空树*/
status ClearBiTree(BiTree &T)
{
    if (T == NULL) // 空树
        return OK;
    ClearBiTree(T->lchild);
    ClearBiTree(T->rchild);
    free(T); // 释放结点空间
    T = NULL;
    return OK;
}

/*判断树的深度*/
int BiTreeDepth(BiTree T)
{
    if (T == NULL) // 空树
        return 0;
    int ldepth = BiTreeDepth(T->lchild);            // 左子树的深度
    int rdepth = BiTreeDepth(T->rchild);            // 右子树的深度
    return (ldepth > rdepth ? ldepth : rdepth) + 1; // max(ldepth,rdepth)+1
}

/*定位结点*/
BiTree LocateNode(BiTree T, KeyType e)
{
    if (T == NULL) // 空树
        return NULL;
    if (T->data.key == e) // 找到结点
        return T;
    BiTree p = LocateNode(T->lchild, e); // 在左子树中查找
    if (p != NULL)
        return p;
    p = LocateNode(T->rchild, e); // 在右子树中查找
    return p;
}

/*赋值*/
status Assign(BiTree &T, KeyType e, TElemType value)
{
    BiTree p = LocateNode(T, e); // 定位结点
    if (p == NULL)
        return INFEASIBLE; // 未找到结点
    else if (value.key != e && LocateNode(T, value.key) != NULL)
        return ERROR; // 检查是否有相同的关键字
    p->data = value;  // 赋值
    return OK;
}

/*获得兄弟结点*/
BiTree GetSibling(BiTree T, KeyType e)
{
    if (T == NULL) // 空树
        return NULL;
    if (T->lchild != NULL && T->lchild->data.key == e) // 找到左孩子
        return T->rchild;
    if (T->rchild != NULL && T->rchild->data.key == e) // 找到右孩子
        return T->lchild;
    BiTree p = GetSibling(T->lchild, e); // 在左子树中查找
    if (p != NULL)
        return p;
    p = GetSibling(T->rchild, e); // 在右子树中查找
    return p;
}

/*插入结点*/
status InsertNode(BiTree &T, KeyType e, int LR, TElemType c)
{
    BiTree p = LocateNode(T, e); // 定位结点
    if (p == NULL)
        return INFEASIBLE;
    else if (LocateNode(T, c.key) != NULL)
        return ERROR; // 检查是否有相同的关键字
    if (LR == -1)     // 插入根结点
    {
        BiTree q = (BiTree)malloc(sizeof(BiTNode));
        q->data = c;
        q->lchild = NULL;
        q->rchild = T;
        T = q;
    }
    if (LR == 0) // 插入左孩子
    {
        BiTree q = (BiTree)malloc(sizeof(BiTNode));
        q->data = c;
        q->lchild = NULL;
        q->rchild = p->lchild;
        p->lchild = q;
    }
    if (LR == 1) // 插入右孩子
    {
        BiTree q = (BiTree)malloc(sizeof(BiTNode));
        q->data = c;
        q->lchild = NULL;
        q->rchild = p->rchild;
        p->rchild = q;
    }
    return OK;
}

/*删除结点*/
status DeleteNode(BiTree &T, KeyType e)
// 1.如关键字为e的结点度为0，删除即可;
// 2.如关键字为e的结点度为1，用关键字为e的结点孩子代替被删除的e位置;
// 3.如关键字为e的结点度为2，用e的左子树(简称为LC)代替被删除的e位置，将e的右子树(简称为RC)作为LC中最右节点的右子树。
// 成功删除结点后返回OK，否则返回ERROR。
{
    BiTree p = LocateNode(T, e); // 定位结点
    if (p == NULL)
        return ERROR; // 未找到结点
    if (p == T)       // 删除根结点
    {
        if (p->lchild == NULL && p->rchild == NULL) // 度为0
        {
            free(p);
            T = NULL;
            return OK;
        }
        if (p->lchild != NULL && p->rchild == NULL) // 度为1
        {
            T = T->lchild;
            free(p);
            return OK;
        }
        if (p->lchild == NULL && p->rchild != NULL) // 度为1
        {
            T = T->rchild;
            free(p);
            return OK;
        }
        if (p->lchild != NULL && p->rchild != NULL) // 度为2
        {
            BiTree q = p->lchild;
            while (q->rchild != NULL)
                q = q->rchild;
            q->rchild = p->rchild;
            T = T->lchild;
            free(p);
            return OK;
        }
    }
    else // 删除非根结点
    {
        if (p->lchild == NULL && p->rchild == NULL)
        {
            BiTree q = GetParent(T, e);
            if (q->lchild != NULL && q->lchild->data.key == e)
                q->lchild = NULL;
            else
                q->rchild = NULL;
            free(p);
            return OK;
        }
        if (p->lchild != NULL && p->rchild == NULL)
        {
            BiTree q = GetParent(T, e);
            if (q->lchild != NULL && q->lchild->data.key == e)
                q->lchild = p->lchild;
            else
                q->rchild = p->lchild;
            free(p);
            return OK;
        }
        if (p->lchild == NULL && p->rchild != NULL)
        {
            BiTree q = GetParent(T, e);
            if (q->lchild != NULL && q->lchild->data.key == e)
                q->lchild = p->rchild;
            else
                q->rchild = p->rchild;
            free(p);
            return OK;
        }
        if (p->lchild != NULL && p->rchild != NULL)
        {
            BiTree q = GetParent(T, e);
            BiTree r = p->lchild;
            while (r->rchild != NULL)
                r = r->rchild;
            r->rchild = p->rchild;
            if (q->lchild != NULL && q->lchild->data.key == e)
                q->lchild = p->lchild;
            else
                q->rchild = p->lchild;
            free(p);
            return OK;
        }
    }
    return OK;
}

/*获得父结点*/
BiTree GetParent(BiTree T, KeyType e)
{
    if (T == NULL) // 空树
        return NULL;
    if (T->lchild != NULL && T->lchild->data.key == e) // 找到左孩子
        return T;
    if (T->rchild != NULL && T->rchild->data.key == e) // 找到右孩子
        return T;
    BiTree p = GetParent(T->lchild, e); // 在左子树中查找
    if (p != NULL)
        return p;
    p = GetParent(T->rchild, e); // 在右子树中查找
    return p;
}

/*先序遍历*/
status PreOrderTraverse(BiTree T, void (*visit)(BiTree))
{
    if (T)
    {
        visit(T);
        PreOrderTraverse(T->lchild, visit);
        PreOrderTraverse(T->rchild, visit);
    }
    return OK;
}

/*中序遍历*/
status InOrderTraverse(BiTree T, void (*visit)(BiTree))
{
    BiTree stack[100];
    int top = 0;
    stack[top++] = T; // 根结点入栈
    while (top)       // 栈不空
    {
        T = stack[top - 1]; // 得到栈顶元素
        while (T)           // 向左下走到尽头
        {
            T = T->lchild;
            stack[top++] = T;
        }
        top--; // 弹出NULL指针
        if (top)
        {
            T = stack[--top]; // 弹出栈顶元素
            visit(T);
            stack[top++] = T->rchild; // 右子树入栈
        }
    }
    return OK;
}

/*后序遍历*/
status PostOrderTraverse(BiTree T, void (*visit)(BiTree))
{
    if (T)
    {
        PostOrderTraverse(T->lchild, visit);
        PostOrderTraverse(T->rchild, visit);
        visit(T);
    }
    return OK;
}

/*层序遍历*/
status LevelOrderTraverse(BiTree T, void (*visit)(BiTree))
{
    BiTree queue[100];
    int front = 0, rear = 0;
    queue[rear++] = T;    // 根结点入队
    while (front != rear) // 队列不空
    {
        T = queue[front++];
        visit(T);
        if (T->lchild != NULL) // 左孩子入队
            queue[rear++] = T->lchild;
        if (T->rchild != NULL) // 右孩子入队
            queue[rear++] = T->rchild;
    }
    return OK;
}

/*保存树*/
status SaveBiTree(TreeList TL)
// 将二叉树的结点数据写入到文件FileName中
{
    FILE *fp = fopen(FileName, "w");
    if (fp == NULL)
        fp = fopen(FileName, "wb");
    // 先序写入到文件
    for (int i = 0; i < TL.length; i++)
    {
        fprintf(fp, "----------\n");                        // 分隔符
        fprintf(fp, "name:%s\n", TL.elem[i].name);          // 树的名字
        fprintf(fp, "length:%d\n", Save_Len(TL.elem[i].T)); // 树的长度
        BiTree T = TL.elem[i].T;                            // 树的根结点
        BiTree stack[100];                                  // 栈
        int top = 0;
        stack[top++] = T;
        while (top)
        {
            T = stack[--top]; // 出栈
            if (T == NULL)
            {
                fprintf(fp, "0 null\n"); // 空结点
                continue;
            }
            fprintf(fp, "%d %s\n", T->data.key, T->data.others); // 结点的关键字和其他信息
            stack[top++] = T->rchild;                            // 右孩子入栈
            stack[top++] = T->lchild;                            // 左孩子入栈
        }
        fprintf(fp, "----------\n");
    }
    fclose(fp);
    return OK;
}

/*读取树*/
status LoadBiTree(TreeList &TL)
// 读入文件FileName的结点数据，创建二叉树
{
    FILE *fp = fopen(FileName, "r");
    if (fp == NULL)
        return ERROR;
    char currrent_name[MAX_NAME_LENGTH]; // 当前树的名字
    TL.length = 0;                       // 初始化树的数量
    while (TL.length < MAX_TREE_NUM && fscanf(fp, "----------\nname:%s\n", currrent_name) != EOF)
    {
        int len;
        fscanf(fp, "length:%d\n", &len);   // 树的长度
        ClearBiTree(TL.elem[TL.length].T); // 清空原有树
        strcpy(TL.elem[TL.length].name, currrent_name);
        TElemType definition[100];
        for (int i = 0; i < len; i++) // 读入树的结点
            fscanf(fp, "%d %s\n", &definition[i].key, definition[i].others);
        definition[len].key = -1;
        int a = 0;
        CreateBiTree(TL.elem[TL.length].T, definition, a);
        fscanf(fp, "----------\n");
        TL.length++; // 树的数量加1
    }
    fclose(fp);
    return OK;
}

/*翻转树*/
status InvertTree(BiTree &T)
//(不能使用中序 会出错)
{
    if (T == NULL)
        return OK;
    // 交换左右子树
    BiTree temp = T->lchild;
    T->lchild = T->rchild;
    T->rchild = temp;
    // 递归翻转左右子树
    InvertTree(T->lchild);
    InvertTree(T->rchild);
    return OK;
}

/*保存树的长度*/
int Save_Len(BiTree T)
{
    if (T == NULL)
        return 1; // 空节点也要保存
    return Save_Len(T->lchild) + Save_Len(T->rchild) + 1;
}

/*求树的最大路径和*/
int MaxPathSum(BiTree T)
{
    // 如果当前结点为空，返回0
    if (T == NULL)
        return 0;

    // // 如果当前结点是叶子结点，直接返回该结点的键值
    // if (T->lchild == NULL && T->rchild == NULL)
    //     return T->data.key;

    // 递归计算左右子树的最大路径和
    int leftmax = 0;
    int rightmax = 0;
    leftmax = MaxPathSum(T->lchild);
    rightmax = MaxPathSum(T->rchild);

    // 返回当前节点的键值加上左右子树的最大路径和中的较大值
    return T->data.key + (leftmax > rightmax ? leftmax : rightmax);
}

/*求树的最近公共祖先*/
BiTree LowestCommonAncestor(BiTree T, KeyType e1, KeyType e2)
{
    if (T == NULL) // 空树
        return NULL;
    if (T->data.key == e1 || T->data.key == e2) // 找到结点
        return T;
    BiTree left = LowestCommonAncestor(T->lchild, e1, e2);  // 递归左子树
    BiTree right = LowestCommonAncestor(T->rchild, e1, e2); // 递归右子树
    if (left != NULL && right != NULL)                      // 左右子树都找到
        return T;
    else if (left != NULL) // 左子树找到
        return left;
    else if (right != NULL) // 右子树找到
        return right;
    return NULL;
}
