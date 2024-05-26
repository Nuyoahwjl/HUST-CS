typedef int KeyType; 
typedef struct {  //二叉树结点数据类型定义
     KeyType  key;
     char others[20];
} TElemType; 
typedef struct BiTNode{  //二叉链表结点的定义
      TElemType  data;
      struct BiTNode *lchild,*rchild;
} BiTNode, *BiTree;


status CreateBiTree(BiTree &T,TElemType definition[])
/*根据带空枝的二叉树先序遍历序列definition构造一棵二叉树，将根节点指针赋值给T并返回OK，
如果有相同的关键字，返回ERROR。此题允许通过增加其它函数辅助实现本关任务*/
{
    static int i=0;
    if(i==0) //第一次调用时检查是否有相同的关键字
    {
        for(int j=0;definition[j].key!=-1;j++)
        {
            for(int k=j+1;definition[k].key!=-1;k++)
            {
                if(definition[j].key==definition[k].key&&definition[k].key!=0) 
                return ERROR;
            }
        }
    }

    if(definition[i].key==-1) return OK; //递归结束条件
        
    if(definition[i].key==0) 
    {
        T=NULL;
        i++;
        return OK;
    }
    else
    {
        T=(BiTree)malloc(sizeof(BiTNode));
        T->data=definition[i];
        i++;
        CreateBiTree(T->lchild,definition);
        CreateBiTree(T->rchild,definition);
    }
    return OK;
}
    

status ClearBiTree(BiTree &T)
//将二叉树设置成空，并删除所有结点，释放结点空间
{
    if(T==NULL) return OK;
    ClearBiTree(T->lchild);
    ClearBiTree(T->rchild);
    free(T);
    T=NULL;
    return OK;
}


int BiTreeDepth(BiTree T)
//求二叉树T的深度
{
    if(T==NULL) return 0;
    int ldepth=BiTreeDepth(T->lchild);
    int rdepth=BiTreeDepth(T->rchild);
    return (ldepth>rdepth?ldepth:rdepth)+1;
}


BiTNode* LocateNode(BiTree T,KeyType e)
//查找结点
{
    if(T==NULL) return NULL;
    if(T->data.key==e) return T;
    BiTNode* p=LocateNode(T->lchild,e);
    if(p!=NULL) return p;
    p=LocateNode(T->rchild,e);
    return p;
}


status Assign(BiTree &T,KeyType e,TElemType value)
//实现结点赋值。此题允许通过增加其它函数辅助实现本关任务
{
    BiTNode* p=LocateNode(T,e);
    if(p==NULL) return ERROR;
    else if(value.key!=e&&LocateNode(T,value.key)!=NULL) 
        return ERROR; // 检查是否有相同的关键字
    p->data=value;
    return OK;
}


BiTNode* GetSibling(BiTree T,KeyType e)
//实现获得兄弟结点
{
    if(T==NULL) return NULL;
    if(T->lchild!=NULL&&T->lchild->data.key==e) return T->rchild;
    if(T->rchild!=NULL&&T->rchild->data.key==e) return T->lchild;
    BiTNode* p=GetSibling(T->lchild,e);
    if(p!=NULL) return p;
    p=GetSibling(T->rchild,e);
    return p;
}


status InsertNode(BiTree &T,KeyType e,int LR,TElemType c)
//插入结点。此题允许通过增加其它函数辅助实现本关任务
{
    BiTNode* p=LocateNode(T,e);
    if(p==NULL) return ERROR;
    else if(LocateNode(T,c.key)!=NULL) 
        return ERROR; // 检查是否有相同的关键字
    if(LR==-1)
    {
        BiTNode* q=(BiTNode*)malloc(sizeof(BiTNode));
        q->data=c;
        q->lchild=NULL;
        q->rchild=T;
        T=q;
    }
    if(LR==0)
    {
        BiTNode* q=(BiTNode*)malloc(sizeof(BiTNode));
        q->data=c;
        q->lchild=NULL;
        q->rchild=p->lchild;
        p->lchild=q;
    }
    if(LR==1)
    {
        BiTNode* q=(BiTNode*)malloc(sizeof(BiTNode));
        q->data=c;
        q->lchild=NULL;
        q->rchild=p->rchild;
        p->rchild=q;
    }
    return OK;
}


status DeleteNode(BiTree &T,KeyType e)
//删除结点。此题允许通过增加其它函数辅助实现本关任务
//1.如关键字为e的结点度为0，删除即可;
//2.如关键字为e的结点度为1，用关键字为e的结点孩子代替被删除的e位置;
//3.如关键字为e的结点度为2，用e的左子树(简称为LC)代替被删除的e位置，将e的右子树(简称为RC)作为LC中最右节点的右子树。
//成功删除结点后返回OK，否则返回ERROR。
{
    BiTNode *p = LocateNode(T, e);
    if (p == NULL)
        return ERROR;
    if (p == T)
    {
        if(p->lchild==NULL&&p->rchild==NULL)
        {
            free(p);
            T=NULL;
            return OK;
        }
        if(p->lchild!=NULL&&p->rchild==NULL)
        {
            T=T->lchild;
            free(p);
            return OK;
        }   
        if(p->lchild==NULL&&p->rchild!=NULL)
        {
            T=T->rchild;
            free(p);
            return OK;
        }
        if(p->lchild!=NULL&&p->rchild!=NULL)
        {
            BiTNode* q=p->lchild;
            while(q->rchild!=NULL) q=q->rchild;
            q->rchild=p->rchild;
            T=T->lchild;
            free(p);
            return OK;
        }
    }
    else
    {
        if (p->lchild == NULL && p->rchild == NULL)
        {
            BiTree q=GetParent(T,e);
            if(q->lchild!=NULL&&q->lchild->data.key==e) q->lchild=NULL;
            else q->rchild=NULL;    
            free(p);
            return OK;
        }
        if (p->lchild != NULL && p->rchild == NULL)
        {
            // BiTNode *q = p->lchild;
            // p->data = q->data;
            // p->lchild = q->lchild;
            // p->rchild = q->rchild;
            // free(q);
            // return OK;
            BiTree q=GetParent(T,e);
            if(q->lchild!=NULL&&q->lchild->data.key==e) q->lchild=p->lchild;
            else q->rchild=p->lchild;   
            free(p);
            return OK;
        }
        if (p->lchild == NULL && p->rchild != NULL)
        {
            // BiTNode *q = p->rchild;
            // p->data = q->data;
            // p->lchild = q->lchild;
            // p->rchild = q->rchild;
            // free(q);
            // return OK;
            BiTree q=GetParent(T,e);
            if(q->lchild!=NULL&&q->lchild->data.key==e) q->lchild=p->rchild;
            else q->rchild=p->rchild;
            free(p);
            return OK;
        }
        if (p->lchild != NULL && p->rchild != NULL)
        {
            // BiTNode *q = p->lchild;
            // while (q->rchild != NULL)
            //     q = q->rchild;
            // q->rchild = p->rchild;
            // p->data = p->lchild->data;
            // p->rchild = p->lchild->rchild;
            // BiTNode *t = p->lchild;
            // p->lchild = p->lchild->lchild;
            // free(t);
            // return OK;
            BiTree q=GetParent(T,e);
            BiTree r=p->lchild;
            while(r->rchild!=NULL) r=r->rchild;
            r->rchild=p->rchild;
            if(q->lchild!=NULL&&q->lchild->data.key==e) q->lchild=p->lchild;
            else q->rchild=p->lchild;
            free(p);
            return OK;
        }
    }
}


BiTNode* GetParent(BiTree T,KeyType e) //获取父节点
{
    if(T==NULL) return NULL;
    if(T->lchild!=NULL&&T->lchild->data.key==e) return T;
    if(T->rchild!=NULL&&T->rchild->data.key==e) return T;
    BiTNode* p=GetParent(T->lchild,e);
    if(p!=NULL) return p;
    p=GetParent(T->rchild,e);
    return p;
}


status PreOrderTraverse(BiTree T,void (*visit)(BiTree))
//先序遍历二叉树T
{
    if(T)
    {
        visit(T);
        PreOrderTraverse(T->lchild,visit);
        PreOrderTraverse(T->rchild,visit);
    }
    return OK;
}


status InOrderTraverse(BiTree T,void (*visit)(BiTree))
//中序遍历二叉树T
{
    BiTree stack[100];
    int top=0;
    stack[top++]=T;
    while(top)
    {
        T=stack[top-1];
        while(T)
        {
            T=T->lchild;
            stack[top++]=T;
        }
        top--;
        if(top)
        {
            T=stack[--top];
            visit(T);   
            stack[top++]=T->rchild;
        }
    }
    return OK;
}


status PostOrderTraverse(BiTree T,void (*visit)(BiTree))
//后序遍历二叉树T
{
    if(T)
    {
        PostOrderTraverse(T->lchild,visit);
        PostOrderTraverse(T->rchild,visit);
        visit(T);
    }
    return OK;
}


status LevelOrderTraverse(BiTree T,void (*visit)(BiTree))
//按层遍历二叉树T
{
    BiTree queue[100];
    int front=0,rear=0;
    queue[rear++]=T;
    while(front!=rear)
    {
        T=queue[front++];
        visit(T);
        if(T->lchild!=NULL) queue[rear++]=T->lchild;
        if(T->rchild!=NULL) queue[rear++]=T->rchild;
    }
    return OK;
}


status SaveBiTree(BiTree T, char FileName[])
//将二叉树的结点数据写入到文件FileName中
{
    FILE *fp=fopen(FileName,"w");
    if(fp==NULL) 
        fp=fopen(FileName,"wb");
    if(T==NULL)
        return ERROR;
    //先序写入到文件
    BiTree stack[100];  
    int top=0;
    stack[top++]=T;
    while(top)
    {
        T=stack[--top];
        if(T==NULL) 
        {
            fprintf(fp,"0 null\n");
            continue;
        }
        fprintf(fp,"%d %s\n",T->data.key,T->data.others);
        stack[top++]=T->rchild;
        stack[top++]=T->lchild;
    }
    fclose(fp);
    return OK;
}
status LoadBiTree(BiTree &T,  char FileName[])
//读入文件FileName的结点数据，创建二叉树
{
    FILE *fp=fopen(FileName,"r");
    if(fp==NULL) 
        return ERROR;
    TElemType definition[100];
    int i=0;
    while(fscanf(fp,"%d %s\n",&definition[i].key,definition[i].others)!=EOF)
    {
        i++;
    }
    definition[i].key=-1;
    CreateBiTree(T,definition);
    fclose(fp);
    return OK;
}
