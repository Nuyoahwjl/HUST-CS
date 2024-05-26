//U
#include <iostream>
using namespace std;
void f(int m, int n){//m为被分解的数，n为位数，r为数值 
    int r;
    if (m == 0) return;//m被分解完 
    r=m%2;
    m=m/2;
    f(m, n+1);
    if (m!=0 && r!=0){//不是第一个数，输出+ 
        cout<<"+";
    }
    if (r == 1){
        if (n == 0) cout<<"2(0)";
        else if (n == 1) cout<<"2";
        else if (n == 2) cout<<"2(2)";
        else{//2的指数大于2继续分解
            cout<<"2(";
            f(n, 0);
            cout<<")";
        }
    }
}
int main(){
    int num;
	cin>>num;
    f(num, 0);
    return 0;
}
