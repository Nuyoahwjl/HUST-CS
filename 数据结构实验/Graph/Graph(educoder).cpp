#include "stdio.h"
#include "stdlib.h"
#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASIBLE -1
#define OVERFLOW -2
#define MAX_VERTEX_NUM 20
typedef int status;
typedef int KeyType;
typedef enum
{
    DG,
    DN,
    UDG,
    UDN
} GraphKind;
typedef struct
{
    KeyType key;
    char others[20];
} VertexType; // 顶点类型定义

typedef struct ArcNode
{                            // 表结点类型定义
    int adjvex;              // 顶点位置编号
    struct ArcNode *nextarc; // 下一个表结点指针
} ArcNode;
typedef struct VNode
{                      // 头结点及其数组类型定义
    VertexType data;   // 顶点信息
    ArcNode *firstarc; // 指向第一条弧
} VNode, AdjList[MAX_VERTEX_NUM];
typedef struct
{                       // 邻接表的类型定义
    AdjList vertices;   // 头结点数组
    int vexnum, arcnum; // 顶点数、弧数
    GraphKind kind;     // 图的类型
} ALGraph;




void visit(VertexType v)
{
    printf(" %d %s",v.key,v.others);
}


status CreateCraph(ALGraph &G,VertexType V[],KeyType VR[][2])
/*根据V和VR构造图T并返回OK，如果V和VR不正确，返回ERROR
如果有相同的关键字，返回ERROR。此题允许通过增加其它函数辅助实现本关任务*/
{
    int vexnum = 0, arcnum = 0;
    // 检查V和VR是否正确
    for(int i=0;V[i].key!=-1;i++)
    {
        vexnum++;
        for(int j=i+1;V[j].key!=-1;j++)
        {
            if(V[i].key==V[j].key) 
            return ERROR; //有相同的关键字
        }
    }
    if(vexnum<=0||vexnum>MAX_VERTEX_NUM) return ERROR;
    for(int i=0;VR[i][0]!=-1;i++)
    {
        arcnum++;
        if(find(V,VR[i][0],vexnum)==ERROR||find(V,VR[i][1],vexnum)==ERROR)
            return ERROR; //VR中有不在V中的顶点
    }
    G.vexnum=vexnum;
    G.arcnum=arcnum;    
    G.kind=UDG;

    // 构造邻接表
    for(int i=0;i<vexnum;i++)
    {
        G.vertices[i].data=V[i];
        G.vertices[i].firstarc=NULL;
    }
    for(int k=0;k<G.arcnum;k++)
    {
        int i,j;
        for(i=0;i<G.vexnum;i++)
        {
            if(G.vertices[i].data.key==VR[k][0]) break;
        }
        for(j=0;j<G.vexnum;j++)
        {
            if(G.vertices[j].data.key==VR[k][1]) break;
        }
        ArcNode *p=(ArcNode*)malloc(sizeof(ArcNode));
        p->adjvex=j;
        p->nextarc=G.vertices[i].firstarc;
        G.vertices[i].firstarc=p;
        ArcNode *q=(ArcNode*)malloc(sizeof(ArcNode));
        q->adjvex=i;
        q->nextarc=G.vertices[j].firstarc;
        G.vertices[j].firstarc=q;
    }
    return OK;

}
status find(VertexType V[],int key,int n)
{
    for(int i=0;i<n;i++)
    {
        if(V[i].key==key) return OK;
    }
    return ERROR;
}


status DestroyGraph(ALGraph &G)
/*销毁无向图G,删除G的全部顶点和边*/
{
    for(int i=0;i<G.vexnum;i++)
    {
        ArcNode *p=G.vertices[i].firstarc;
        while(p!=NULL)
        {
            ArcNode *q=p;
            p=p->nextarc;
            free(q);
        }
    }
    G.vexnum=0;
    G.arcnum=0;
    return OK;
}


int LocateVex(ALGraph G,KeyType u)
//根据u在图G中查找顶点，查找成功返回位序，否则返回-1；
{
    for(int i=0;i<G.vexnum;i++)
    {
        if(G.vertices[i].data.key==u) return i;
    }
    return -1;
}


status PutVex(ALGraph &G,KeyType u,VertexType value)
//根据u在图G中查找顶点，查找成功将该顶点值修改成value，返回OK；
//如果查找失败或关键字不唯一，返回ERROR
{
    int i=LocateVex(G,u);
    if(i==-1) return ERROR;
    for(int j=0;j<G.vexnum;j++)
    {
        if(G.vertices[j].data.key==value.key&&j!=i) return ERROR;
    }
    G.vertices[i].data=value;
    return OK;
}


int FirstAdjVex(ALGraph G,KeyType u)
//根据u在图G中查找顶点，查找成功返回顶点u的第一邻接顶点位序，否则返回-1；
{
    int i=LocateVex(G,u);
    if(i==-1) return -1;
    if(G.vertices[i].firstarc==NULL) return -1;
    return G.vertices[i].firstarc->adjvex;
}


int NextAdjVex(ALGraph G,KeyType v,KeyType w)
//v对应G的一个顶点,w对应v的邻接顶点；操作结果是返回v的（相对于w）下一个邻接顶点的位序；如果w是最后一个邻接顶点，或v、w对应顶点不存在，则返回-1。
{
    int i=LocateVex(G,v);
    int j=LocateVex(G,w);
    if(i==-1||j==-1) return -1;
    ArcNode *p=G.vertices[i].firstarc;
    while(p!=NULL&&p->adjvex!=j)
    {
        p=p->nextarc;
    }
    if(p==NULL||p->nextarc==NULL) return -1;
    return p->nextarc->adjvex;
}


status InsertVex(ALGraph &G,VertexType v)
//在图G中插入顶点v，成功返回OK,否则返回ERROR
{
    if(G.vexnum>=MAX_VERTEX_NUM) return ERROR;
    for(int i=0;i<G.vexnum;i++)
    {
        if(G.vertices[i].data.key==v.key) return ERROR;
    }
    G.vertices[G.vexnum].data=v;
    G.vertices[G.vexnum].firstarc=NULL;
    G.vexnum++;
    return OK;
}


status DeleteVex(ALGraph &G,KeyType v)
//在图G中删除关键字v对应的顶点以及相关的弧，成功返回OK,否则返回ERROR
{
    // 判断图中是否只有一个顶点
    if(G.vexnum == 1)
    {
        // printf("图中只有一个顶点，不能删除\n");
        return ERROR;
    }
    // 查找顶点
    int i=LocateVex(G,v);
    if(i==-1) return ERROR;
    // 删除顶点
    ArcNode *p=G.vertices[i].firstarc;
    while(p!=NULL)
    {
        ArcNode *q=p;
        p=p->nextarc;
        free(q);
    }
    for(int j=i;j<G.vexnum-1;j++)
    {
        G.vertices[j]=G.vertices[j+1]; // 位序前移
    }
    G.vexnum--;
    // 删除相关弧
    for (int j = 0; j < G.vexnum; ++j)
    {
        ArcNode *p = G.vertices[j].firstarc;
        ArcNode *q = NULL;
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
                G.arcnum--;
            }
            else
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


status InsertArc(ALGraph &G,KeyType v,KeyType w)
//在图G中增加弧<v,w>，成功返回OK,否则返回ERROR
{
    int i=LocateVex(G,v);
    int j=LocateVex(G,w);
    if(i==-1||j==-1) return ERROR;
    ArcNode *temp=G.vertices[i].firstarc;
    while(temp!=NULL) 
    {
        if(temp->adjvex==j)
            return ERROR;
        temp=temp->nextarc;
    }
    ArcNode *p=(ArcNode*)malloc(sizeof(ArcNode));
    p->adjvex=j;
    p->nextarc=G.vertices[i].firstarc;
    G.vertices[i].firstarc=p;
    ArcNode *q=(ArcNode*)malloc(sizeof(ArcNode));
    q->adjvex=i;
    q->nextarc=G.vertices[j].firstarc;
    G.vertices[j].firstarc=q;
    G.arcnum++;
    return OK;
}


status DeleteArc(ALGraph &G,KeyType v,KeyType w)
//在图G中删除弧<v,w>，成功返回OK,否则返回ERROR
{
    // 判断节点是否存在
    int i=LocateVex(G,v);
    int j=LocateVex(G,w);
    if(i==-1||j==-1) return ERROR;
    int flag;
    ArcNode *p=G.vertices[i].firstarc;
    ArcNode *q=NULL;
    while(p!=NULL)
    {
        if(p->adjvex==j)
        {
            flag=1;
            if(q==NULL)
                G.vertices[i].firstarc=p->nextarc;
            else 
                q->nextarc=p->nextarc;
            ArcNode *temp=p;
            p=p->nextarc;
            free(temp);
            G.arcnum--;
        }
        else
        {
            q=p;
            p=p->nextarc;
        }
    }
    p=G.vertices[j].firstarc;
    q=NULL;
    while(p!=NULL)
    {
        if(p->adjvex==i)
        {
            if(q==NULL)
                G.vertices[j].firstarc=p->nextarc;
            else 
                q->nextarc=p->nextarc;
            ArcNode *temp=p;
            p=p->nextarc;
            free(temp);
        }
        else
        {
            q=p;
            p=p->nextarc;
        }
    }
    if(flag) return OK;
    return ERROR;
}


status DFSTraverse(ALGraph &G,void (*visit)(VertexType))
//对图G进行深度优先搜索遍历，依次对图中的每一个顶点使用函数visit访问一次，且仅访问一次
{
    bool visited[G.vexnum];
    for(int v=0;v<G.vexnum;v++)
        visited[v]=false;
    for(int v=0;v<G.vexnum;v++)
    {
        if(!visited[v])
        DFS(G,v,visited,visit);
    }
    return OK;
}
void DFS(ALGraph G,int v,bool visited[],void (*visit)(VertexType))
{
    visit(G.vertices[v].data);
    visited[v]=true;
    for(int w=FirstAdjVex(G,G.vertices[v].data.key);w>=0;w=NextAdjVex(G,G.vertices[v].data.key,G.vertices[w].data.key))
    {
        if(!visited[w])
            DFS(G,w,visited,visit);
    }
}


status BFSTraverse(ALGraph &G,void (*visit)(VertexType))
//对图G进行广度优先搜索遍历，依次对图中的每一个顶点使用函数visit访问一次，且仅访问一次
{
    bool visited[G.vexnum];
    for(int v=0;v<G.vexnum;v++)
        visited[v]=false;
    int Q[100];
    int front=0;
    int rear=0;
    for(int v=0;v<G.vexnum;v++)
    {
        if(!visited[v])
        {
            Q[rear++]=v;
            visited[v]=true;
            visit(G.vertices[v].data);
            int u;
            while (front != rear)
            {
                u = Q[front++];
                for(int w=FirstAdjVex(G,G.vertices[u].data.key);w>=0;w=NextAdjVex(G,G.vertices[u].data.key,G.vertices[w].data.key))
                {
                    if(!visited[w])
                    {
                        Q[rear++] = w;
                        visited[w]=true;
                        visit(G.vertices[w].data);
                    }
                }
            }
        }
    }
    return OK;
}
//出队时访问
// status BFSTraverse2(ALGraph &G,void (*visit)(VertexType))
// //对图G进行广度优先搜索遍历，依次对图中的每一个顶点使用函数visit访问一次，且仅访问一次
// {
//     bool visited[G.vexnum];
//     for(int v=0;v<G.vexnum;v++)
//         visited[v]=false;
//     int Q[100];
//     int front=0;
//     int rear=0;
//     for(int v=0;v<G.vexnum;v++)
//     {
//         if(!visited[v])
//         {
//             Q[rear++]=v;
//             int u;
//             while (front != rear)
//             {
//                 u = Q[front++];
//                 if(!visited[u])
//                 {
//                     visit(G.vertices[u].data);
//                     visited[u]=true;
//                 }
//                 for(int w=FirstAdjVex(G,G.vertices[u].data.key);w>=0;w=NextAdjVex(G,G.vertices[u].data.key,G.vertices[w].data.key))
//                 {
//                     if(!visited[w])
//                     {
//                         Q[rear++] = w;
//                     }
//                 }
//             }
//         }
//     }
//     return OK;
// }


status SaveGraph(ALGraph G, char FileName[])
//将图的数据写入到文件FileName中
{
    FILE *fp=fopen(FileName,"w");
    if(fp==NULL) 
        fp=fopen(FileName,"wb");
    if(G.vexnum<=0||G.vexnum>MAX_VERTEX_NUM)
        return ERROR;
    fprintf(fp,"vexnum:%d arcnum:%d kind:%d\n",G.vexnum,G.arcnum,G.kind);
    for(int i=0;i<G.vexnum;i++)
    {
        fprintf(fp,"%d %s\n",G.vertices[i].data.key,G.vertices[i].data.others);
    }
    for(int i=0;i<G.vexnum;i++)
    {
        ArcNode *p=G.vertices[i].firstarc;
        while(p!=NULL)
        {
            fprintf(fp,"%d %d\n",G.vertices[i].data.key,G.vertices[p->adjvex].data.key);
            p=p->nextarc;
        }
    }
    fclose(fp);
}
status LoadGraph(ALGraph &G, char FileName[])
//读入文件FileName的图数据，创建图的邻接表
{
    FILE *fp=fopen(FileName,"r");
    if(fp==NULL) 
        fp=fopen(FileName,"rb");
    if(fp==NULL)
        return ERROR;
    DestroyGraph(G);
    fscanf(fp,"vexnum:%d arcnum:%d kind:%d\n",&G.vexnum,&G.arcnum,&G.kind);
    for(int i=0;i<G.vexnum;i++)
    {
        fscanf(fp,"%d %s\n",&G.vertices[i].data.key,G.vertices[i].data.others);
        G.vertices[i].firstarc=NULL;
    }
    int VR[100][2];
    for(int i=2*G.arcnum-1;i>=0;i--)
    {
        fscanf(fp,"%d %d\n",&VR[i][0],&VR[i][1]);
    }
    for(int k=0;k<2*G.arcnum;k++)
    {
        int i,j;
        for(i=0;i<G.vexnum;i++)
        {
            if(G.vertices[i].data.key==VR[k][0]) break;
        }
        for(j=0;j<G.vexnum;j++)
        {
            if(G.vertices[j].data.key==VR[k][1]) break;
        }
        ArcNode *p=(ArcNode*)malloc(sizeof(ArcNode));
        p->adjvex=j;
        p->nextarc=G.vertices[i].firstarc;
        G.vertices[i].firstarc=p;
    }
    fclose(fp);
    return OK;
}


// 头歌上没有G.arcnum
// int getnum(Graph G)
// {
//     int num=0;
//     for(int i=0;i<G.vexnum;i++)
//     {
//         ArcNode *p=G.vertices[i].firstarc;
//         while(p!=NULL)
//         {
//             num++;
//             p=p->nextarc;
//         }
//     }
//     return num/2;
// }

// status SaveGraph(ALGraph G, char FileName[])
// //将图的数据写入到文件FileName中
// {
//     FILE *fp=fopen(FileName,"w");
//     if(fp==NULL) 
//         fp=fopen(FileName,"wb");
//     if(G.vexnum<=0||G.vexnum>MAX_VERTEX_NUM)
//         return ERROR;
//     G.arcnum=getnum(G);
//     fprintf(fp,"vexnum:%d arcnum:%d kind:%d\n",G.vexnum,G.arcnum,G.kind);
//     for(int i=0;i<G.vexnum;i++)
//     {
//         fprintf(fp,"%d %s\n",G.vertices[i].data.key,G.vertices[i].data.others);
//     }
//     for(int i=0;i<G.vexnum;i++)
//     {
//         ArcNode *p=G.vertices[i].firstarc;
//         while(p!=NULL)
//         {
//             fprintf(fp,"%d %d\n",G.vertices[i].data.key,G.vertices[p->adjvex].data.key);
//             p=p->nextarc;
//         }
//     }
//     fclose(fp);
// }

// status LoadGraph(ALGraph &G, char FileName[])
// //读入文件FileName的图数据，创建图的邻接表
// {
//     FILE *fp=fopen(FileName,"r");
//     if(fp==NULL) 
//         fp=fopen(FileName,"rb");
//     if(fp==NULL)
//         return ERROR;
//     DestroyGraph(G);
//     fscanf(fp,"vexnum:%d arcnum:%d kind:%d\n",&G.vexnum,&G.arcnum,&G.kind);
//     for(int i=0;i<G.vexnum;i++)
//     {
//         fscanf(fp,"%d %s\n",&G.vertices[i].data.key,G.vertices[i].data.others);
//         G.vertices[i].firstarc=NULL;
//     }
//     int VR[100][2];
//     for(int i=2*G.arcnum-1;i>=0;i--)
//     {
//         fscanf(fp,"%d %d\n",&VR[i][0],&VR[i][1]);
//     }
//     for(int k=0;k<2*G.arcnum;k++)
//     {
//         int i,j;
//         for(i=0;i<G.vexnum;i++)
//         {
//             if(G.vertices[i].data.key==VR[k][0]) break;
//         }
//         for(j=0;j<G.vexnum;j++)
//         {
//             if(G.vertices[j].data.key==VR[k][1]) break;
//         }
//         ArcNode *p=(ArcNode*)malloc(sizeof(ArcNode));
//         p->adjvex=j;
//         p->nextarc=G.vertices[i].firstarc;
//         G.vertices[i].firstarc=p;
//     }
//     fclose(fp);
//     return OK;
// }

