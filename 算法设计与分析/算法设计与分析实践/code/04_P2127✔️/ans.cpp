#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <climits> // 必须包含此头文件才能使用 INT_MAX

using namespace std;

int main() 
{
    int N;
    cin >> N;
    vector<long long> arr(N + 1), sorted(N + 1);
    unordered_map<long long, int> index_map;
    vector<bool> visited(N + 1, false);
    
    for (int i = 1; i <= N; i++)    
    {
        cin >> arr[i];
        sorted[i] = arr[i];
    }

    sort(sorted.begin() + 1, sorted.end());
    
    // 构建索引映射
    for (int i = 1; i <= N; i++)
        index_map[sorted[i]] = i;

    long long global_min = sorted[1]; // 全局最小值
    long long total_cost = 0;

    for (int i = 1; i <= N; i++) 
    {
        if (visited[i] || arr[i] == sorted[i])
            // 跳过已访问或已就位的元素
            continue;

        int cycle_size = 0; // 置换环大小
        long long cycle_sum = 0; // 置换环总和
        long long cycle_min = 2147483647; // 置换环最小值
        int current_index = i; // 当前索引

        // 处理一个环
        while (!visited[current_index]) {
            visited[current_index] = true;
            cycle_size++;
            cycle_sum += arr[current_index];
            cycle_min = min(cycle_min, static_cast<long long>(arr[current_index]));
            current_index = index_map[arr[current_index]];
        }

        // 通过环内最小元素计算代价
        long long cycle_cost = cycle_min * (cycle_size - 1) + cycle_sum - cycle_min;

        // 通过全局最小元素计算代价
        long long global_cost = (global_min + cycle_min) * 2 + global_min * (cycle_size - 1) + cycle_sum - cycle_min;

        total_cost += min(cycle_cost, global_cost);
    }

    cout << total_cost;
    return 0;
}
