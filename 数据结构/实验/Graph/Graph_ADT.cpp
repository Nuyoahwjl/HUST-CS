#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASIBLE -1
#define OVERFLOW -2
#define FileName "data.txt"
#define MAX_VERTEX_NUM 20
#define MAX_NAME_LENGTH 20
#define MAX_GRAPH_NUM 10
typedef int status;
typedef int KeyType;
// 图的类型定义
typedef enum
{
    DG,
    DN,
    UDG,
    UDN
} GraphKind;
// 顶点类型定义
typedef struct
{
    KeyType key;
    char others[20];
} VertexType;
// 表结点类型定义
typedef struct ArcNode
{
    int adjvex;              // 顶点位置编号
    struct ArcNode *nextarc; // 下一个表结点指针
} ArcNode;
// 头结点及其数组类型定义
typedef struct VNode
{
    VertexType data;   // 顶点信息
    ArcNode *firstarc; // 指向第一条弧
} VNode, AdjList[MAX_VERTEX_NUM];
// 邻接表的类型定义
typedef struct
{
    AdjList vertices;   // 头结点数组
    int vexnum, arcnum; // 顶点数、弧数
    GraphKind kind;     // 图的类型
} ALGraph;
// 图的集合类型定义
typedef struct
{
    struct
    {
        ALGraph G;
        char name[MAX_NAME_LENGTH];
    } elem[MAX_GRAPH_NUM];
    int length; // 图的数量
} GraphList;
GraphList GL; // 全局变量，图的集合

/*函数声明*/
void PrintMenu1();                                                     // 打印主菜单
void PrintMenu2();                                                     // 打印子菜单
void main2(ALGraph &G);                                                // 子页面
status CreateCraph(ALGraph &G, VertexType V[], KeyType VR[][2]);       // 创建图
status find(VertexType V[], int key, int n);                           // 查找关键字
int SelectGraph(GraphList GL, char name[]);                            // 选择图
status DestroyGraph(ALGraph &G);                                       // 销毁图
status SaveGraph(GraphList GL);                                        // 保存图
status LoadGraph(GraphList &G);                                        // 读取图
void visit(VertexType v);                                              // 访问函数
int LocateVex(ALGraph G, KeyType u);                                   // 定位顶点
status PutVex(ALGraph &G, KeyType u, VertexType value);                // 修改顶点
int FirstAdjVex(ALGraph G, KeyType u);                                 // 第一个邻接顶点
int NextAdjVex(ALGraph G, KeyType v, KeyType w);                       // 下一个邻接顶点
status InsertVex(ALGraph &G, VertexType v);                            // 插入顶点
status DeleteVex(ALGraph &G, KeyType v);                               // 删除顶点
status InsertArc(ALGraph &G, KeyType v, KeyType w);                    // 插入弧
status DeleteArc(ALGraph &G, KeyType v, KeyType w);                    // 删除弧
status DFSTraverse(ALGraph &G, void (*visit)(VertexType));             // 深度优先搜索
void DFS(ALGraph G, int v, bool visited[], void (*visit)(VertexType)); // 深度优先搜索辅助函数
status BFSTraverse(ALGraph &G, void (*visit)(VertexType));             // 广度优先搜索
status VerticesSetLessThanK(ALGraph G, KeyType v, int k, int *result); // 与顶点v距离小于k的顶点集合
int ShortestPathLength(ALGraph G, KeyType v, KeyType w);               // 顶点v到顶点w的最短路径长度
int ConnectedComponentsNums(ALGraph G);                                // 连通分量数

/*主函数*/
int main()
{
    PrintMenu1();  // 打印主菜单
    GL.length = 0; // 初始化图的数量
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
        case 1: // 创建图
        {
            printf("|-------------------------------------------------------|\n"
                   "|----------You can create a total of %d graphs.---------|\n",
                   MAX_GRAPH_NUM); // 提示用户可以创建的树的数量
            if (GL.length > 1)     // 提示用户当前已经创建的树的数量
                printf("|--------Currently, %d graphs have been created.---------|\n", GL.length);
            else
                printf("|----------Currently, %d graph has been created.---------|\n", GL.length);
            printf("|-------------------------------------------------------|\n\n");
            printf("Please input the name of the graph: ");
            scanf("%s", GL.elem[GL.length].name);
            printf("Please input the key and others of the vertexes:(end with -1 nil)\n");
            VertexType V[MAX_VERTEX_NUM + 1]; // 顶点数组
            KeyType VR[100][2];               // 弧数组
            int i = 0;
            do
            {
                scanf("%d%s", &V[i].key, V[i].others);
            } while (V[i++].key != -1);
            printf("Please input the key of the arcs:(end with -1 -1)\n");
            i = 0;
            do
            {
                scanf("%d%d", &VR[i][0], &VR[i][1]);
            } while (VR[i++][0] != -1);
            if (CreateCraph(GL.elem[GL.length].G, V, VR) == OK) // 创建图
            {
                printf("Create graph successfully!\n");
                GL.length++; // 图的数量加一
            }
            else
                printf("Create graph failed!\n");
            break;
        }
        case 2: // 删除图
        {
            printf("Please input the name of the graph you want to delete:\n");
            char delete_name[MAX_NAME_LENGTH]; // 删除图的名字
            scanf("%s", delete_name);
            if (SelectGraph(GL, delete_name) == -1) // 未找到图
            {
                printf("ERROR:There is no such graph!\n");
                break;
            }
            else
            {
                DestroyGraph(GL.elem[SelectGraph(GL, delete_name)].G); // 销毁图
                for (int i = SelectGraph(GL, delete_name); i < GL.length - 1; i++)
                {
                    GL.elem[i] = GL.elem[i + 1]; // 前移
                }
                GL.length--; // 图的数量减一
                printf("The graph has been deleted successfully!\n");
            }
            break;
        }
        case 3: // 选择图
        {
            printf("Please input the name of the tree you want to select:\n");
            char select_name[MAX_NAME_LENGTH];
            scanf("%s", select_name);
            int loc = SelectGraph(GL, select_name); // 选择图的位置
            if (loc == -1)                          // 未找到图
            {
                printf("ERROR:There is no such graph!\n");
                break;
            }
            main2(GL.elem[loc].G); // 调用子页面
            PrintMenu1();          // 从子页面跳回时再次打印主菜单
            break;
        }
        case 4: // 保存图
        {
            if (GL.length == 0) // 没有图可保存
            {
                printf("There is no graph to save.\n");
                break;
            }
            if (SaveGraph(GL) == OK)
                printf("Save successfully!\n");
            else
                printf("Save failed!\n");
            break;
        }
        case 5: // 读取图
        {
            if (LoadGraph(GL) == OK)
                printf("Load successfully!\n");
            else
                printf("Load failed!\n");
            break;
        }
        case 6: // 显示所有图
        {
            if (GL.length == 0) // 没有图可显示
            {
                printf("There is no graph to show.\n");
                break;
            }
            for (int i = 0; i < GL.length; i++) // 显示所有图的名字
            {
                printf("%s\n", GL.elem[i].name);
            }
            break;
        }
        case 7: // 查找图
        {
            printf("Please input the name of the graph you want to search:\n");
            char search_name[MAX_NAME_LENGTH];
            scanf("%s", search_name);
            if (SelectGraph(GL, search_name) != -1) // 找到图
                // 输出图的位置
                printf("The location of the graph is %d.\n", SelectGraph(GL, search_name) + 1);
            else
                printf("There is no graph named %s.\n", search_name);
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
void main2(ALGraph &G)
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
        case 1: // 销毁图
        {
            if (DestroyGraph(G) == OK)
                printf("Destroy graph successfully!\n");
            else
                printf("Destroy graph failed!\n");
            break;
        }
        case 2: // 定位顶点
        {
            KeyType u;
            printf("Please input the key of the vertex:\n");
            scanf("%d", &u);
            int loc = LocateVex(G, u);
            if (loc == -1) // 顶点不存在
                printf("The vertex does not exist.\n");
            else // 输出顶点位置
                printf("The location of the vertex is %d.\n", loc);
            break;
        }
        case 3: // 修改顶点
        {
            KeyType u;
            VertexType value;
            printf("Please input the key of the vertex you want to modify:\n");
            scanf("%d", &u);
            printf("Please input the new key and others of the vertex:\n");
            scanf("%d%s", &value.key, value.others);
            if (PutVex(G, u, value) == OK)
                printf("Modify vertex successfully!\n");
            else
                printf("Modify vertex failed!\n"); // 修改顶点失败
            break;
        }
        case 4: // 第一个邻接顶点
        {
            KeyType u;
            printf("Please input the key of the vertex:\n");
            scanf("%d", &u);
            int loc = FirstAdjVex(G, u);
            if (loc == -1) // 顶点不存在或没有邻接顶点
                printf("The vertex does not exist or has no adjacent vertex.\n");
            else // 输出第一个邻接顶点的位置和关键字
            {
                printf("The location of the first adjvex is %d.\n", loc);
                printf("The key of the first adjvex is %d.\n", G.vertices[loc].data.key);
            }
            break;
        }
        case 5: // 下一个邻接顶点
        {
            KeyType v, w;
            printf("Please input the key of the vertex and its adjacent vertex:\n");
            scanf("%d%d", &v, &w); // 输入顶点和邻接顶点
            int loc = NextAdjVex(G, v, w);
            if (loc == -1) // 顶点不存在或没有邻接顶点
                printf("The vertex does not exist or has no adjacent vertex.\n");
            else // 输出下一个邻接顶点的位置和关键字
            {
                printf("The location of the next adjvex is %d.\n", loc);
                printf("The key of the next adjvex is %d.\n", G.vertices[loc].data.key);
            }
            break;
        }
        case 6: // 插入顶点
        {
            VertexType v;
            printf("Please input the key and others of the vertex:\n");
            scanf("%d%s", &v.key, v.others);
            if (InsertVex(G, v) == OK)
                printf("Insert vertex successfully!\n");
            else
                printf("Insert vertex failed!\n");
            break;
        }
        case 7: // 删除顶点
        {
            KeyType v;
            printf("Please input the key of the vertex you want to delete:\n");
            scanf("%d", &v);
            if (DeleteVex(G, v) == OK)
                printf("Delete vertex successfully!\n");
            else
                printf("Delete vertex failed!\n");
            break;
        }
        case 8: // 插入弧
        {
            KeyType v, w;
            printf("Please input the key of the vertex and its adjacent vertex:\n");
            scanf("%d%d", &v, &w);
            if (InsertArc(G, v, w) == OK)
                printf("Insert arc successfully!\n");
            else
                printf("Insert arc failed!\n");
            break;
        }
        case 9: // 删除弧
        {
            KeyType v, w;
            printf("Please input the key of the vertex and its adjacent vertex:\n");
            scanf("%d%d", &v, &w);
            if (DeleteArc(G, v, w) == OK)
                printf("Delete arc successfully!\n");
            else
                printf("Delete arc failed!\n");
            break;
        }
        case 10: // 深度优先遍历
        {
            DFSTraverse(G, visit);
            break;
        }
        case 11: // 广度优先遍历
        {
            BFSTraverse(G, visit);
            break;
        }
        case 12: // 与顶点v距离小于k的顶点集合
        {
            printf("Please input the key of the vertex you want to query:\n");
            KeyType v; // 顶点v
            scanf("%d", &v);
            printf("Please input k:\n");
            int k; // 距离
            scanf("%d", &k);
            int result[MAX_GRAPH_NUM];             // 结果数组
            VerticesSetLessThanK(G, v, k, result); // 与顶点v距离小于k的顶点集合
            for (int i = 0; i < MAX_GRAPH_NUM; i++)
            {
                if (result[i] != -1)
                {
                    printf("%d,%s\n", G.vertices[result[i]].data.key, G.vertices[result[i]].data.others);
                }
                else
                    break;
            }
            break;
        }
        case 13: // 顶点v到顶点w的最短路径长度
        {
            KeyType v, w;
            printf("Please input the key of the two vertexs:\n");
            scanf("%d%d", &v, &w);
            int length = ShortestPathLength(G, v, w);
            if (length == -1)
                printf("The vertex does not exist or has no path.\n");
            else
                printf("The shortest path length is %d.\n", length);
            break;
        }
        case 14: // 连通分量数
        {
            int num = ConnectedComponentsNums(G);
            printf("The number of connected components is %d.\n", num);
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
    printf("|===============Menu for multiple Graphs================|\n");
    printf("|-------------------------------------------------------|\n");
    printf("|              1.    Create a Graph                     |\n");
    printf("|              2.    Delete a Graph                     |\n");
    printf("|              3. Select a single Graph                 |\n");
    printf("|              4. Save All Data To File                 |\n");
    printf("|              5. Load All Data From File               |\n");
    printf("|              6.    Show All Graph                     |\n");
    printf("|              7.    Search a Graph                     |\n");
    printf("|              0.         EXIT                          |\n");
    printf("|=======================================================|\n\n");
}

/*子菜单*/
void PrintMenu2()
{
    printf("|================Menu for single Graph==================|\n");
    printf("|-------------------------------------------------------|\n");
    printf("|  1.  DestroyGraph        2.  LocateVex                |\n");
    printf("|  3.  PutVex              4.  FirstAdjVex              |\n");
    printf("|  5.  NextAdjVex          6.  InsertVex                |\n");
    printf("|  7.  DeleteVex           8.  InsertArc                |\n");
    printf("|  9.  DeleteArc           10. DFSTraverse              |\n");
    printf("|  11. BFSTraverse         12. VerticesSetLessThanK     |\n");
    printf("|  13. ShortestPathLength  14. ConnectedComponentsNums  |\n");
    printf("|                     0. EXIT                           |\n");
    printf("|=======================================================|\n\n");
}

/*创建图*/
status CreateCraph(ALGraph &G, VertexType V[], KeyType VR[][2])
/*根据V和VR构造图T并返回OK，如果V和VR不正确，返回ERROR
如果有相同的关键字，返回ERROR。此题允许通过增加其它函数辅助实现本关任务*/
{
    // 头插法
    int vexnum = 0, arcnum = 0;
    // 检查V和VR是否正确
    for (int i = 0; V[i].key != -1; i++)
    {
        vexnum++;
        for (int j = i + 1; V[j].key != -1; j++)
        {
            if (V[i].key == V[j].key)
                return ERROR; // 有相同的关键字
        }
    }
    if (vexnum <= 0 || vexnum > MAX_VERTEX_NUM)
        return ERROR; // V不正确
    for (int i = 0; VR[i][0] != -1; i++)
    {
        arcnum++;
        if (find(V, VR[i][0], vexnum) == ERROR || find(V, VR[i][1], vexnum) == ERROR)
            return ERROR; // VR中有不在V中的顶点
    }
    G.vexnum = vexnum;
    G.arcnum = arcnum;
    G.kind = UDG;

    // 构造邻接表
    for (int i = 0; i < vexnum; i++)
    {
        G.vertices[i].data = V[i];     // 顶点信息
        G.vertices[i].firstarc = NULL; // 边表头指针初始化为空
    }
    for (int k = 0; k < G.arcnum; k++)
    {
        int i, j;
        for (i = 0; i < G.vexnum; i++) // 找到VR[k][0]的位置
        {
            if (G.vertices[i].data.key == VR[k][0])
                break;
        }
        for (j = 0; j < G.vexnum; j++) // 找到VR[k][1]的位置
        {
            if (G.vertices[j].data.key == VR[k][1])
                break;
        }
        // 插入边<v,w>
        ArcNode *p = (ArcNode *)malloc(sizeof(ArcNode));
        p->adjvex = j;
        p->nextarc = G.vertices[i].firstarc;
        G.vertices[i].firstarc = p;
        // 插入边<w,v>
        ArcNode *q = (ArcNode *)malloc(sizeof(ArcNode));
        q->adjvex = i;
        q->nextarc = G.vertices[j].firstarc;
        G.vertices[j].firstarc = q;
    }
    return OK;
}

/*创建图时查找弧的关键字是否存在*/
status find(VertexType V[], int key, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (V[i].key == key)
            return OK; // 找到
    }
    return ERROR;
}

/*选择图*/
int SelectGraph(GraphList GL, char name[])
{
    for (int i = 0; i < GL.length; i++)
    {
        if (strcmp(GL.elem[i].name, name) == 0)
            return i; // 找到图
    }
    return -1; // 未找到图
}

/*销毁图*/
status DestroyGraph(ALGraph &G)
/*销毁无向图G,删除G的全部顶点和边*/
{
    for (int i = 0; i < G.vexnum; i++)
    {
        ArcNode *p = G.vertices[i].firstarc;
        while (p != NULL) // 删除边表结点
        {
            ArcNode *q = p;
            p = p->nextarc;
            free(q);
        }
    }
    G.vexnum = 0; // 顶点数置为0
    G.arcnum = 0; // 边数置为0
    return OK;
}

/*保存图*/
status SaveGraph(GraphList GL)
// 将图的数据写入到文件FileName中
{
    FILE *fp = fopen(FileName, "w");
    if (fp == NULL)
        fp = fopen(FileName, "wb");
    for (int i = 0; i < GL.length; i++)
    {
        fprintf(fp, "--------------------\n");
        fprintf(fp, "name:%s\n", GL.elem[i].name);       // 图名
        fprintf(fp, "vexnum:%d\n", GL.elem[i].G.vexnum); // 顶点数
        fprintf(fp, "arcnum:%d\n", GL.elem[i].G.arcnum); // 边数
        for (int j = 0; j < GL.elem[i].G.vexnum; j++)    // 顶点信息
        {
            fprintf(fp, "%d %s\n", GL.elem[i].G.vertices[j].data.key, GL.elem[i].G.vertices[j].data.others);
        }
        for (int j = 0; j < GL.elem[i].G.vexnum; j++) // 边信息
        {
            ArcNode *p = GL.elem[i].G.vertices[j].firstarc;
            while (p != NULL)
            {
                fprintf(fp, "%d %d\n", GL.elem[i].G.vertices[j].data.key, GL.elem[i].G.vertices[p->adjvex].data.key);
                p = p->nextarc;
            }
        }
        fprintf(fp, "--------------------\n");
    }
    fclose(fp);
    return OK;
}

/*读取图*/
status LoadGraph(GraphList &GL)
// 读入文件FileName的图数据，创建图的邻接表
{
    FILE *fp = fopen(FileName, "r");
    if (fp == NULL)
        fp = fopen(FileName, "rb");
    if (fp == NULL)
        return ERROR;
    for (int i = 0; i < GL.length; i++)
    {
        DestroyGraph(GL.elem[i].G); // 销毁当前GL中的所有图（覆盖读入）
    }
    GL.length = 0;
    while (fscanf(fp, "--------------------\n") != EOF)
    {
        fscanf(fp, "name:%s\n", GL.elem[GL.length].name);
        fscanf(fp, "vexnum:%d\n", &GL.elem[GL.length].G.vexnum);
        fscanf(fp, "arcnum:%d\n", &GL.elem[GL.length].G.arcnum);
        for (int i = 0; i < GL.elem[GL.length].G.vexnum; i++) // 读入顶点信息
        {
            fscanf(fp, "%d %s\n", &GL.elem[GL.length].G.vertices[i].data.key, GL.elem[GL.length].G.vertices[i].data.others);
            GL.elem[GL.length].G.vertices[i].firstarc = NULL;
        }
        int VR[100][2];
        // 逆序读入边信息
        for (int i = 2 * GL.elem[GL.length].G.arcnum - 1; i >= 0; i--)
        {
            fscanf(fp, "%d %d\n", &VR[i][0], &VR[i][1]);
        }
        // 构造邻接表（头插法）
        for (int k = 0; k < 2 * GL.elem[GL.length].G.arcnum; k++)
        {
            int i, j;
            for (i = 0; i < GL.elem[GL.length].G.vexnum; i++)
            {
                if (GL.elem[GL.length].G.vertices[i].data.key == VR[k][0])
                    break;
            }
            for (j = 0; j < GL.elem[GL.length].G.vexnum; j++)
            {
                if (GL.elem[GL.length].G.vertices[j].data.key == VR[k][1])
                    break;
            }
            ArcNode *p = (ArcNode *)malloc(sizeof(ArcNode));
            p->adjvex = j;
            p->nextarc = GL.elem[GL.length].G.vertices[i].firstarc;
            GL.elem[GL.length].G.vertices[i].firstarc = p;
        }
        GL.length++;
        fscanf(fp, "--------------------\n");
    }
    fclose(fp);
    return OK;
}

/*访问函数*/
void visit(VertexType v)
{
    printf(" %d %s", v.key, v.others);
}

/*定位顶点*/
int LocateVex(ALGraph G, KeyType u)
// 根据u在图G中查找顶点，查找成功返回位序，否则返回-1；
{
    for (int i = 0; i < G.vexnum; i++)
    {
        if (G.vertices[i].data.key == u)
            return i; // 找到
    }
    return -1;
}

/*修改顶点*/
status PutVex(ALGraph &G, KeyType u, VertexType value)
// 根据u在图G中查找顶点，查找成功将该顶点值修改成value，返回OK；
// 如果查找失败或关键字不唯一，返回ERROR
{
    int i = LocateVex(G, u);
    if (i == -1)
        return ERROR;
    for (int j = 0; j < G.vexnum; j++)
    {
        if (G.vertices[j].data.key == value.key && j != i)
            return ERROR; // 关键字不唯一
    }
    G.vertices[i].data = value;
    return OK;
}

/*第一个邻接顶点*/
int FirstAdjVex(ALGraph G, KeyType u)
// 根据u在图G中查找顶点，查找成功返回顶点u的第一邻接顶点位序，否则返回-1；
{
    int i = LocateVex(G, u);
    if (i == -1)
        return -1;
    if (G.vertices[i].firstarc == NULL)
        return -1;                         // 没有邻接顶点
    return G.vertices[i].firstarc->adjvex; // 返回第一个邻接顶点的位序
}

/*下一个邻接顶点*/
int NextAdjVex(ALGraph G, KeyType v, KeyType w)
// v对应G的一个顶点,w对应v的邻接顶点；操作结果是返回v的（相对于w）下一个邻接顶点的位序；如果w是最后一个邻接顶点，或v、w对应顶点不存在，则返回-1。
{
    int i = LocateVex(G, v);
    int j = LocateVex(G, w);
    if (i == -1 || j == -1)
        return -1; // 顶点不存在
    ArcNode *p = G.vertices[i].firstarc;
    while (p != NULL && p->adjvex != j)
    {
        p = p->nextarc;
    }
    if (p == NULL || p->nextarc == NULL)
        return -1;             // w是最后一个邻接顶点
    return p->nextarc->adjvex; // 返回下一个邻接顶点的位序
}

/*插入顶点*/
status InsertVex(ALGraph &G, VertexType v)
// 在图G中插入顶点v，成功返回OK,否则返回ERROR
{
    if (G.vexnum >= MAX_VERTEX_NUM)
        return ERROR; // 顶点数已满
    for (int i = 0; i < G.vexnum; i++)
    {
        if (G.vertices[i].data.key == v.key)
            return ERROR; // 关键字重复
    }
    G.vertices[G.vexnum].data = v;        // 插入顶点
    G.vertices[G.vexnum].firstarc = NULL; // 边表头指针初始化为空
    G.vexnum++;                           // 顶点数加1
    return OK;
}

/*删除顶点*/
status DeleteVex(ALGraph &G, KeyType v)
// 在图G中删除关键字v对应的顶点以及相关的弧，成功返回OK,否则返回ERROR
{
    // 判断图中是否只有一个顶点
    if (G.vexnum == 1)
    {
        // printf("图中只有一个顶点，不能删除\n");
        return ERROR;
    }
    // 查找顶点
    int i = LocateVex(G, v);
    if (i == -1)
        return ERROR; //
    // 删除顶点
    ArcNode *p = G.vertices[i].firstarc;
    while (p != NULL)
    {
        ArcNode *q = p;
        p = p->nextarc;
        free(q);
    }
    for (int j = i; j < G.vexnum - 1; j++)
    {
        G.vertices[j] = G.vertices[j + 1]; // 位序前移
    }
    G.vexnum--;
    // 删除相关弧
    for (int j = 0; j < G.vexnum; ++j)
    {
        ArcNode *p = G.vertices[j].firstarc;
        ArcNode *q = NULL;
        while (p != NULL)
        {
            if (p->adjvex == i) // 删除弧<v,w>
            {
                if (q == NULL)
                    G.vertices[j].firstarc = p->nextarc;
                else
                    q->nextarc = p->nextarc;
                ArcNode *temp = p;
                p = p->nextarc;
                free(temp);
                G.arcnum--; // 边数减一
            }
            else // 顶点位序前移
            {
                if (p->adjvex > i)
                    p->adjvex--;
                q = p;
                p = p->nextarc;
            }
        }
    }
    return OK;
}

/*插入弧*/
status InsertArc(ALGraph &G, KeyType v, KeyType w)
// 在图G中增加弧<v,w>，成功返回OK,否则返回ERROR
{
    int i = LocateVex(G, v);
    int j = LocateVex(G, w);
    if (i == -1 || j == -1)
        return ERROR; // 顶点不存在
    ArcNode *temp = G.vertices[i].firstarc;
    while (temp != NULL)
    {
        if (temp->adjvex == j) // 弧已存在
            return ERROR;
        temp = temp->nextarc;
    }
    // 插入弧<v,w>
    ArcNode *p = (ArcNode *)malloc(sizeof(ArcNode));
    p->adjvex = j;
    p->nextarc = G.vertices[i].firstarc;
    G.vertices[i].firstarc = p;
    // 插入弧<w,v>
    ArcNode *q = (ArcNode *)malloc(sizeof(ArcNode));
    q->adjvex = i;
    q->nextarc = G.vertices[j].firstarc;
    G.vertices[j].firstarc = q;
    G.arcnum++; // 边数加1
    return OK;
}

/*删除弧*/
status DeleteArc(ALGraph &G, KeyType v, KeyType w)
// 在图G中删除弧<v,w>，成功返回OK,否则返回ERROR
{
    // 判断节点是否存在
    int i = LocateVex(G, v);
    int j = LocateVex(G, w);
    if (i == -1 || j == -1)
        return ERROR;
    int flag = 0;
    // 删除弧<v,w>
    ArcNode *p = G.vertices[i].firstarc;
    ArcNode *q = NULL;
    while (p != NULL)
    {
        if (p->adjvex == j)
        {
            flag = 1;      // 找到弧
            if (q == NULL) // 删除第一个结点
                G.vertices[i].firstarc = p->nextarc;
            else
                q->nextarc = p->nextarc;
            ArcNode *temp = p;
            p = p->nextarc;
            free(temp);
            G.arcnum--;
        }
        else
        {
            q = p;
            p = p->nextarc;
        }
    }
    // 删除弧<w,v>
    p = G.vertices[j].firstarc;
    q = NULL;
    while (p != NULL)
    {
        if (p->adjvex == i)
        {
            if (q == NULL)
                G.vertices[j].firstarc = p->nextarc;
            else
                q->nextarc = p->nextarc;
            ArcNode *temp = p;
            p = p->nextarc;
            free(temp);
        }
        else
        {
            q = p;
            p = p->nextarc;
        }
    }
    if (flag)
        return OK; // 找到弧
    return ERROR;
}

/*深度优先搜索*/
status DFSTraverse(ALGraph &G, void (*visit)(VertexType))
// 对图G进行深度优先搜索遍历，依次对图中的每一个顶点使用函数visit访问一次，且仅访问一次
{
    bool visited[G.vexnum];
    for (int v = 0; v < G.vexnum; v++) // 初始化
        visited[v] = false;
    for (int v = 0; v < G.vexnum; v++) // 深度优先搜索遍历
    {
        if (!visited[v])
            DFS(G, v, visited, visit);
    }
    return OK;
}
void DFS(ALGraph G, int v, bool visited[], void (*visit)(VertexType))
{
    visit(G.vertices[v].data); // 访问顶点v
    visited[v] = true;         // 标记为已访问
    // 访问v的所有未访问邻接顶点
    for (int w = FirstAdjVex(G, G.vertices[v].data.key); w >= 0; w = NextAdjVex(G, G.vertices[v].data.key, G.vertices[w].data.key))
    {
        if (!visited[w])
            DFS(G, w, visited, visit);
    }
}

/*广度优先搜索*/
status BFSTraverse(ALGraph &G, void (*visit)(VertexType))
// 对图G进行广度优先搜索遍历，依次对图中的每一个顶点使用函数visit访问一次，且仅访问一次
{
    bool visited[G.vexnum];
    for (int v = 0; v < G.vexnum; v++)
        visited[v] = false;
    int Q[100];
    int front = 0;
    int rear = 0;
    for (int v = 0; v < G.vexnum; v++)
    {
        if (!visited[v]) // 未访问过
        {
            Q[rear++] = v;
            visited[v] = true;
            visit(G.vertices[v].data);
            int u;
            while (front != rear) // 广度优先搜索遍历
            {
                u = Q[front++];
                // 访问u的所有未访问邻接顶点
                for (int w = FirstAdjVex(G, G.vertices[u].data.key); w >= 0; w = NextAdjVex(G, G.vertices[u].data.key, G.vertices[w].data.key))
                {
                    if (!visited[w])
                    {
                        Q[rear++] = w;
                        visited[w] = true;
                        visit(G.vertices[w].data);
                    }
                }
            }
        }
    }
    return OK;
}

/*与顶点v距离小于k的顶点集合*/
status VerticesSetLessThanK(ALGraph G, KeyType v, int k, int *result)
{
    int num = 0; // 记录结果个数
    int i = LocateVex(G, v);
    if (i == -1)
        return ERROR;       // 顶点不存在
    bool visited[G.vexnum]; // 记录是否访问过
    visited[i] = true;      // 顶点v已访问
    for (int j = 0; j < G.vexnum; j++)
        visited[j] = false;
    int Q[100];
    int front = 0;
    int rear = 0;
    Q[rear++] = i;
    for (int level = 1; level < k; level++) // 循环k-1次
    {
        int last = rear;      // 记录上一层的最后一个顶点位置
        while (front != last) // 广度优先搜索遍历
        {
            int u = Q[front++];
            for (int w = FirstAdjVex(G, G.vertices[u].data.key); w >= 0; w = NextAdjVex(G, G.vertices[u].data.key, G.vertices[w].data.key))
            {
                if (!visited[w]) // 未访问过
                {
                    Q[rear++] = w;
                    visited[w] = true;
                    result[num++] = w; // 记录结果
                }
            }
        }
    }
    result[num] = -1;
    return OK;
}

/*顶点v到顶点w的最短路径长度*/
int ShortestPathLength(ALGraph G, KeyType v, KeyType w)
{
    int i = LocateVex(G, v);
    int j = LocateVex(G, w);
    if (i == -1 || j == -1)
        return -1;
    int dist[G.vexnum]; // 记录最短路径长度
    for (int k = 0; k < G.vexnum; k++)
        dist[k] = -1; // -1表示未找到最短路径
    dist[i] = 0;      // 顶点v到自身的距离为0
    int Q[100];
    int front = 0;
    int rear = 0;
    Q[rear++] = i; // 顶点v入队
    while (front != rear)
    {
        int u = Q[front++]; // 顶点u出队
        for (int k = FirstAdjVex(G, G.vertices[u].data.key); k >= 0; k = NextAdjVex(G, G.vertices[u].data.key, G.vertices[k].data.key))
        {
            if (dist[k] == -1) // -1表示未找到最短路径
            {
                dist[k] = dist[u] + 1; // 更新最短路径长度
                Q[rear++] = k;         // 顶点k入队
            }
        }
    }
    return dist[j]; // 返回i到j的最短路径长度
}

/*连通分量数*/
int ConnectedComponentsNums(ALGraph G)
{
    int num = 0; // 连通分量数
    bool visited[G.vexnum];
    for (int v = 0; v < G.vexnum; v++)
        visited[v] = false;
    for (int v = 0; v < G.vexnum; v++) // 广度优先搜索遍历
    {
        if (!visited[v])
        {
            num++; // 连通分量数加一
            visited[v] = true;
            int Q[100];
            int front = 0;
            int rear = 0;
            Q[rear++] = v;
            while (front != rear)
            {
                int u = Q[front++];
                for (int w = FirstAdjVex(G, G.vertices[u].data.key); w >= 0; w = NextAdjVex(G, G.vertices[u].data.key, G.vertices[w].data.key))
                {
                    if (!visited[w])
                    {
                        Q[rear++] = w;
                        visited[w] = true;
                    }
                }
            }
        }
    }
    return num;
}

