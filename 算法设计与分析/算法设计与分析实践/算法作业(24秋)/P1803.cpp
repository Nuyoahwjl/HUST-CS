#include <bits/stdc++.h>
using namespace std;

int main()
{
    int n;
    cin >> n;
    vector<pair<int, int>> v(n+1);
    for (int i = 1; i <= n; i++)
    {
        cin >> v[i].first >> v[i].second;
    }
    sort(v.begin(), v.end(), [](pair<int, int> a, pair<int, int> b) {
        return a.second < b.second;
    });
    int t=v[1].second;
    int num=1;
    for(int i=2;i<=n;i++){
        if(v[i].first>=t){
            t=v[i].second;
            num++;
        }
    }
    cout<<num<<endl;
    return 0;
}