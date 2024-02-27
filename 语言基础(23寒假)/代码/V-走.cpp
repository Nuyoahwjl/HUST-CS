//V
#include<bits/stdc++.h>
using namespace std;
#define maxn 100005
#define maxm 2000005
#define inf 0x3f3f3f3f
template <typename Tp>
void read(Tp &x){
    char c=getchar();x=0;int f=1;
    while(c<'0'||c>'9'){if(c=='-')f=-1;c=getchar();}
    while(c>='0'&&c<='9'){x=x*10+c-'0';c=getchar();}x*=f;
}//快速读入，不解释 
struct Edge{
    int f,t,w,nxt;
}edge[maxm];
int head[maxn],etot=1;//这里有一个小技巧，存图时边数初值设为一个奇数 
void add_edge(int f,int t,int w){//这样可以利用位运算成对变化找到反向边 
    edge[++etot]=(Edge){f,t,w,head[f]};
    head[f]=etot;
}//链式前向星存图 
//--------以下内容为 zkw线段树
//主要思路，用线段树维护dis
//dis1数组表示在线段树中的dis
//tr数组记录当前最小dis对应的节点编号 
//有关zkw线段树，可以参考洛谷日报的讲解，这里不多说 
int tr[maxn<<2],dis1[maxn<<2],bt;
int n,m,S,T;
void build(){
    for(bt=1;bt<=n+1;bt<<=1);//bt初始化，zkw线段树的初始操作 
    for(int i=1;i<=n;i++)tr[i+bt]=i;//tr数组初始化 
    memset(dis1,0x3f,sizeof(dis1));//dis1数组初始化 
    //因为这里dis初值都是inf，所以可以这样直接赋值 
}
void modify(int x,int val){
    dis1[x]=val;x+=bt;//单点修改 
    for(x>>=1;x;x>>=1){//以下是zkw线段树常规操作 
        if(dis1[tr[x<<1]]<dis1[tr[(x<<1)|1]])tr[x]=tr[x<<1];
        else tr[x]=tr[(x<<1)|1];
    }
}//其实上面的内容并不是很长，只是注释比较多 
int dis[maxn];
void dijkstra(){
    memset(dis,0x3f,sizeof(dis));
    build();//build()不可忘 
    dis[S]=0;modify(S,0);//源点更新 
    int x,y,w;
    for(int j=1;j<=n;j++){//这里tr[1]维护的是[1,n]dis的最小值的节点编号，所以直接调用 
        x=tr[1];modify(x,inf);//这里将x设为极大值，来取代删除操作 
        for(int i=head[x];i;i=edge[i].nxt){
            y=edge[i].t;w=edge[i].w;
            if(dis[y]>dis[x]+w){//dijkstra松弛操作 
                dis[y]=dis[x]+w;
                modify(y,dis[y]);//直接更新 
            }
        }
    }
}
int dx[]={0,1,0,-1,1,1,-1,-1,0,2,0,-2};//12方向及魔法代价 
int dy[]={1,0,-1,0,1,-1,1,-1,2,0,-2,0};
int dw[]={0,0,0,0,2,2,2,2,2,2,2,2};
int a[105][105],cnt[105][105];
struct node{
    int x,y,c;
}b[maxn];
//a存储棋盘上格子的颜色 
//cnt表示棋盘上的格子对应的节点编号 
void preprocess(){//建图 
    int x,y,c,xx,yy,ww;
    for(int i=1;i<=n;i++){
        x=b[i].x;y=b[i].y;c=b[i].c;
        for(int j=0;j<12;j++){
            xx=x+dx[j];yy=y+dy[j];ww=dw[j];
            if(a[xx][yy]){
                if(a[xx][yy]!=c)ww++;
                add_edge(i,cnt[xx][yy],ww);
            }
        }
    }//这一段在上文题解中讲的比较详细，这里不再多说 
    S=cnt[1][1];
}
int main(){
    int mm,x,y,c;
    read(mm);read(n);
    for(int i=1;i<=n;i++){
        read(x);read(y);read(c);
        a[x][y]=c+1;cnt[x][y]=i;
        b[i]=(node){x,y,c+1};
    }//这里c+1，为了方便区分无色格子 
    preprocess();
    dijkstra();//因为在图论中m常代表的含义是边数 
    if(!a[mm][mm]){//所以用mm取代原题目中的m，即棋盘大小 
        int ans=min(dis[cnt[mm][mm-1]],dis[cnt[mm-1][mm]])+2;
        if(ans>=inf)puts("-1");
        else printf("%d\n",ans);
    }//(m,m)没有颜色的特判 
    else{
        if(dis[cnt[mm][mm]]==inf)puts("-1");
        else printf("%d\n",dis[cnt[mm][mm]]);
    }
    return 0;
}
