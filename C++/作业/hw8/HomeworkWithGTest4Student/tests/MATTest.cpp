#include <gtest/gtest.h>
#include "../include/ch13_homework/MAT.h"
#include <iostream>

//用于初始化MAT对象
int a1[2][3] = {
    {1,2,3},
    {4,5,6}
};

int a2[2][3] = {
    {1,1,1},
    {2,2,2}
};

int a3[3][2] = {
    {1,2},
    {3,4},
    {5,6}
};

//a1的转置
int a1trans[3][2] = {
    {1,4},
    {2,5},
    {3,6}
};

//a1+a2的结果
int a1plusa2[2][3] = {
    {2,3,4},
    {6,7,8}
};

//a1 * a3的结果
int a1multia3[2][2] = {
    {22,28},
    {49,64}
};

TEST(TEST_MAT, MAT_test_all){
    //given
    MAT<int> m1(2,3);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 3; j++){
            m1[i][j] = a1[i][j];
        }
    }
    MAT<int> m2(2,3);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 3; j++){
            m2[i][j] = a2[i][j];
        }
    }
    MAT<int> m3(3,2);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 2; j++){
            m3[i][j] = a3[i][j];
        }
    }

    MAT<int> m1trans(3,2);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 2; j++){
            m1trans[i][j] = a1trans[i][j];
        }
    }
    ASSERT_EQ((~m1).toString(),m1trans.toString());  //测试转置

    MAT<int> m1plusm2(2,3);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 3; j++){
            m1plusm2[i][j] = a1plusa2[i][j];
        }
    }
    ASSERT_EQ((m1+m2).toString(),m1plusm2.toString());  //测试+

    MAT<int> m1multim3(2,2);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            m1multim3[i][j] = a1multia3[i][j];
        }
    }
    ASSERT_EQ((m1*m3).toString(),m1multim3.toString());  //测试+

    //测试析构
    int r = 0;
    m1.~MAT();
    m1.~MAT();
    r = 1;
    ASSERT_EQ(1, r);
}