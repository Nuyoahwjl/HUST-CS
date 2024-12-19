#include <iostream>
#include <cstdio>
#include <algorithm>

using namespace std;

// 定义结构体 road，表示一条公路
struct road
{
    int i, j; // 公路连接的两个村庄
    int t;    // 公路修复完成的时间
};

// 比较函数，用于按公路修复完成时间排序
bool compare(road a, road b)
{
    return a.t < b.t; // 按照修复完成时间升序排序
}

int main()
{
    int N, M; // N 表示村庄数，M 表示公路数
    cin >> N >> M; // 输入村庄数和公路数
    struct road r[M]; // 定义一个数组存储所有公路信息
    for(int i = 0; i < M; i++)
    {
        // 输入每条公路的信息
        cin >> r[i].i >> r[i].j >> r[i].t;
    }
    // 按照公路修复完成时间升序排序
    sort(r, r + M, compare);

    int parent[N + 1]; // 并查集的父节点数组
    int rank[N + 1];   // 并查集的秩数组
    for(int i = 0; i <= N; i++)
    {
        parent[i] = i; // 初始化父节点为自身
        rank[i] = 1;   // 初始化秩为 1
    }

    int ans = 0; // 记录最早的通车时间
    int flag = 0; // 标记是否所有村庄都连通
    for(int i = 0; i < M; i++)
    {
        int x = r[i].i; // 获取公路连接的第一个村庄
        int y = r[i].j; // 获取公路连接的第二个村庄
        // 查找 x 的根节点
        while(x != parent[x])
        {
            x = parent[x];
        }
        // 查找 y 的根节点
        while(y != parent[y])
        {
            y = parent[y];
        }
        // 如果 x 和 y 不在同一个集合中，则合并
        if(x != y)
        {
            // 按秩合并
            if(rank[x] > rank[y])
            {
                parent[y] = x;
                rank[x] += rank[y];
            }
            else
            {
                parent[x] = y;
                rank[y] += rank[x];
            }
            ans = r[i].t; // 更新最早通车时间
            // 如果所有村庄都连通，输出结果并退出
            if(rank[x] == N || rank[y] == N)
            {
                cout << ans;
                flag = 1;
                break;
            }
        }
    }
    // 如果遍历完所有公路后仍有村庄不连通，输出 -1
    if(flag == 0)
    {
        cout << "-1";
    }
    return 0;
}