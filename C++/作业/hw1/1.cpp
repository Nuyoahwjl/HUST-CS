#include <iostream>
using namespace std;
int main(){
    int a,b;
    cout << "Enter two numbers(a&b): "<<endl;
    cin >> a >> b;
    if(a>b){
        while(a>=b){
            cout << a << " ";
            a--;
        }
    }
    else{
        while(a<=b){
            cout << a << " ";
            a++;
        }
    }
    return 0;
}