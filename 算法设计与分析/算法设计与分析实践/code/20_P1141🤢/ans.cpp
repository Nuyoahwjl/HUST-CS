#include <iostream>
#include <vector>
#include <queue>
using namespace std;

// 定义迷宫的最大边长，用于预分配数组空间
const int MAXN = 1000;

// 定义方向数组 dx 和 dy，用于描述上下左右移动
const int dx[4] = {-1, 1, 0, 0}; // 上、下、左、右在 x 方向上的增量
const int dy[4] = {0, 0, -1, 1}; // 上、下、左、右在 y 方向上的增量

// 全局变量
int n, m;                       // n 表示迷宫大小，m 表示查询数量
char maze[MAXN][MAXN];          // 存储迷宫网格，'0' 或 '1'
int region[MAXN][MAXN];         // 每个格子的区域编号，-1 表示未访问
int regionSize[MAXN * MAXN];    // 每个区域的大小

// 检查是否可以移动到目标格子
bool isValid(int x, int y, int prevValue) {
    // 条件1：坐标在迷宫范围内
    // 条件2：目标格子未被访问
    // 条件3：目标格子与当前格子值不同（即 '0' 到 '1' 或 '1' 到 '0'）
    return x >= 0 && x < n && y >= 0 && y < n && region[x][y] == -1 && maze[x][y] != prevValue;
}

// 使用 Flood Fill 算法对连通区域进行标记
void floodFill(int startX, int startY, int regionId) {
    queue<pair<int, int>> q;            // 定义队列用于 BFS
    q.push({startX, startY});           // 将起点加入队列
    region[startX][startY] = regionId;  // 标记起点的区域编号
    int size = 0;                       // 初始化当前区域的大小

    // BFS 遍历整个连通区域
    while (!q.empty()) {
        // 获取队首元素的坐标
        int x = q.front().first;
        int y = q.front().second;
        q.pop();
        size++; // 每访问一个格子，区域大小加 1

        // 遍历当前格子的四个方向
        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i]; // 新的 x 坐标
            int ny = y + dy[i]; // 新的 y 坐标
            // 如果可以移动到目标格子，执行以下操作
            if (isValid(nx, ny, maze[x][y])) {
                region[nx][ny] = regionId; // 标记目标格子为当前区域
                q.push({nx, ny});         // 将目标格子加入队列
            }
        }
    }

    // 将计算得到的区域大小记录下来
    regionSize[regionId] = size;
}

int main() {
    // 提高输入输出效率
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 读取迷宫大小 n 和查询数量 m
    cin >> n >> m;

    // 初始化迷宫与区域数组
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> maze[i][j];  // 读取迷宫的每一格字符
            region[i][j] = -1;  // 初始时所有格子区域编号设为 -1，表示未访问
        }
    }

    int currentRegionId = 0; // 当前区域编号，从 0 开始

    // 遍历迷宫，对每个未访问的格子进行 Flood Fill
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (region[i][j] == -1) {              // 如果当前格子未被访问
                floodFill(i, j, currentRegionId);  // 从该格子出发进行 Flood Fill
                currentRegionId++;                // 增加区域编号
            }
        }
    }

    // 处理每个查询
    while (m--) {
        int x, y;
        cin >> x >> y; // 读取查询的格子坐标（1-based 索引）
        --x; // 转换为 0-based 索引
        --y;
        cout << regionSize[region[x][y]] << "\n"; // 输出该格子所属区域的大小
    }

    return 0;
}

