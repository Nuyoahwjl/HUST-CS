//T
//#include <iostream>
//#include <map>
//#include <vector>
//#include <set>
//using namespace std;
//
//set<int> ans;
//map<int, vector<int> > in;
//
//void find(int i)
//{
//	int len = in[i].size();
//	ans.insert(i);
//	for (int j = len - 1; j >= 0; --j)
//    	find(in[i][j]);
//}
//
//int main()
//{
//	int N;
//	cin >> N;
//	for (int i = 1; i <= N; ++i)
//	{
//		int Ci;
//		cin >> Ci;
// 		in[i].resize(Ci);
//		while (Ci--)
//		{
// 			cin >> in[i][Ci];
//		}
//	}	
//	find(1);
//	cout << ans.size();
//}

//优化后的代码使用了一个 visited 数组来记录节点
//是否已经被访问过。同时，使用一个 stack 来模拟
//递归的过程，实现深度优先搜索。每次从栈中取出一
//个节点进行处理，将其标记为已访问，并将其子节点
//入栈。这样可以避免函数调用带来的开销。
#include <iostream>
#include <vector>
#include <set>
using namespace std;

const int MAXN = 1000005;

vector<int> in[MAXN];
bool visited[MAXN];

int main()
{
    int N;
    cin >> N;
    for (int i = 1; i <= N; ++i)
    {
        int Ci;
        cin >> Ci;
        in[i].resize(Ci);
        for (int j = 0; j < Ci; ++j)
        {
            cin >> in[i][j];
        }
    }

    set<int> ans;
    vector<int> stack;
    stack.push_back(1);
    while (!stack.empty())
    {
        int node = stack.back();
        stack.pop_back();
        if (!visited[node])
        {
            visited[node] = true;
            ans.insert(node);
            for (int i = in[node].size() - 1; i >= 0; --i)
            {
                stack.push_back(in[node][i]);
            }
        }
    }

    cout << ans.size();

    return 0;
}
