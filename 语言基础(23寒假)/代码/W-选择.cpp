//W
#include<iostream>
#include<algorithm>
using namespace std;
int a[100] = { 0 };
int n, k;
int sum = 0;
int ans = 0;
bool f(int sum)//判断质数
{
	
	if (sum == 2)
		return true;
	for (int i = 2; i < sum; i++)
	{
		if (sum%i == 0)
		{
			
			return false;
		}
	}
	
	
		return true;
	
 
}
void dfs(int m,int sum ,int z)//传第一个数组下标，和选k个数，m是记录选了几个数
{
	
	if (m == k)//一旦已经选好了三个数就必须回溯
	{
		
		if (f(sum))
		{
			ans++;
		}
		return;//如果不在这个地方回溯的话，还会继续执行下面的语句m的值就会出错 return控制在哪个地方回溯
	}
	//没进入递归之前就是把选的第一个数都循环一遍，也是递归终止的条件
	for (int i = z; i <= n; i++)//进入第二个递归后就相当于把要选的第二个数全部都循环了
	{
		
		
		dfs(m + 1, sum + a[i], i + 1);
	}
	
	return;
}
int main()
{
	
	cin >> n >> k;
	
	for (int i = 1; i <= n; i++)
	{
		cin >> a[i];//要保证是升序排放在数组中
	}
	sort(a,a + n+1);
	dfs(0,0,1);
	cout << ans;
	return 0;
}
