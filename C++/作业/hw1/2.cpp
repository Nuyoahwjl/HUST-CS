#include <iostream>
using namespace std;
int main (){
    int n,temp;
    cout<<"Please enter the counts of integers you want to sum: "<<endl;
    cin>>n;
    int sum=0;
    for(int i=0;i<n;++i){
        cout<<"Please enter the integer(No."<<i+1<<"): ";
        cin>>temp;
        sum+=temp;
    }
    cout<<"The sum of the integers is: "<<sum<<endl;
    return 0;
}