#include <gtest/gtest.h>

#include "../include/ch6_ch8_ch11_homework/DUALQUEUE.h"

#include <iostream>

//测试DUALQUEUE的实例化、将元素加入队列尾部以及toString函数
TEST(TEST_DUALQUEUE, DUALQUEUE_instantiate_append_toString){
    //given
    DUALQUEUE dq(5);
    dq << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 9 << 10;

    //when
    //得到对象queue的字符串表示
    std::string content = dq.toString();

    //then
    std::string target("1 2 3 4 5 6 7 8 9 10");
    ASSERT_EQ(target,content);
}

/*
    测试DUALQUEUE的拷贝构造函数
*/
TEST(TEST_DUALQUEUE, DUALQUEUE_Test_CopyConstructor){
    //given
    DUALQUEUE dqa(5);
    dqa << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素

    //检测内容是否相等
    //when
    DUALQUEUE dqb(dqa);
    //then
    ASSERT_EQ(dqa.toString(),dqb.toString());

    //检查是否为深拷贝
    //when
    dqa.~DUALQUEUE();  //析构对象qa
    //then
    std::string target("1 2 3 4 5");
    ASSERT_EQ(target,dqb.toString()); //这时对象qb的内容应该还在，深拷贝
}

/**

    测试DUALQUEUE的copy assignment（拷贝=）
*/
TEST(TEST_DUALQUEUE, DUALQUEUE_Test_CopyAssignment){
    //given
    DUALQUEUE dqa(1);
    DUALQUEUE dqb(5);
    dqb << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素

    //检测内容是否相等
    //when
    dqa = dqb;
    //then
    ASSERT_EQ(dqa.toString(),dqb.toString());

    //检查是否为深拷贝
    //when
    dqb.~DUALQUEUE();
    //then
    std::string target("1 2 3 4 5");
    ASSERT_EQ(target,dqa.toString()); 
}

/*
    测试DUALQUEUE的移动构造函数
*/
TEST(TEST_DUALQUEUE, DUALQUEUE_Test_MoveConstructor){
    //given
    DUALQUEUE dqa(5);
    dqa << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素
    const int * const inQueenPtr = dqa.getInQueueElemsPtr();  //保留dqa的内部入队列的指针
    const int * const outQueenPtr = dqa.getOutQueueElemsPtr();  //保留dqa的内部出队列的指针

    //when
    DUALQUEUE dqb(std::move(dqa));

    //检测内容是否相等
    std::string target("1 2 3 4 5");
    //then
    ASSERT_EQ(target,dqb.toString());

    //检查是否为浅拷贝
    //then
    ASSERT_EQ(inQueenPtr,dqb.getInQueueElemsPtr());
    ASSERT_EQ(outQueenPtr,dqb.getOutQueueElemsPtr());

    //检查dqa对象是不是为可以安全析构
    //顺便检查了operator int()
    //then
    ASSERT_EQ(0,dqa);
    ASSERT_EQ(0,dqa.capacity());
    ASSERT_EQ(nullptr,dqa.getInQueueElemsPtr());
    ASSERT_EQ(nullptr,dqa.getOutQueueElemsPtr());
}

/*
    测试DUALQUEUE的move assignment（移动=）
*/
TEST(TEST_DUALQUEUE, DUALQUEUE_Test_MoveAssignment){
    //given
    DUALQUEUE dqa(1);
    DUALQUEUE dqb(5);
    dqb << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素
    const int * const inQueenPtr = dqb.getInQueueElemsPtr();    //保留dqb的内部入队列的指针
    const int * const outQueenPtr = dqb.getOutQueueElemsPtr();  //保留dqb的内部出队列的指针

    //when
    dqa = std::move(dqb);

    //检测内容是否相等
    std::string target("1 2 3 4 5");
    //then
    ASSERT_EQ(target,dqa.toString());

    //检查是否为浅拷贝
    //then
    ASSERT_EQ(inQueenPtr,dqa.getInQueueElemsPtr());
    ASSERT_EQ(outQueenPtr,dqa.getOutQueueElemsPtr());

    //检查dqb对象是不是为可以安全析构
    //顺便检查了operator int()
    //then
    ASSERT_EQ(0,dqb);
    ASSERT_EQ(0,dqb.capacity());
    ASSERT_EQ(nullptr,dqb.getInQueueElemsPtr());
    ASSERT_EQ(nullptr,dqb.getOutQueueElemsPtr());

}

/*
    测试DUALQUEUE的析构函数
*/
TEST(TEST_DUALQUEUE, DUALQUEUE_Test_Deconstructor){
    //given
    DUALQUEUE dqa(5);
    dqa << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素
    int rtn = 0;

    //when
    dqa.~DUALQUEUE();
    dqa.~DUALQUEUE();
    rtn = 1;    //如果析构函数实现正确，会执行到rtn = 1;

    //then
    ASSERT_EQ(0,dqa);
    ASSERT_EQ(nullptr,dqa.getInQueueElemsPtr());
    ASSERT_EQ(nullptr,dqa.getOutQueueElemsPtr());
    ASSERT_EQ(1, rtn);

}

/*
    测试DUALQUEUE的类型转换和capacity函数
*/
TEST(TEST_DUALQUEUE, DUALQUEUE_Test_TypeCast_Capacity){
    //given
    DUALQUEUE dqa(5);

    //then1
    ASSERT_EQ(0,dqa);
    ASSERT_EQ(10,dqa.capacity());

    //when
    dqa << 1 << 2 << 3 << 4 << 5 << 6;  //队列加入六个元素

    //then2
    ASSERT_EQ(6,dqa);
    ASSERT_EQ(10,dqa.capacity());
}

//测试DUALQUEUE的实例化、将元素加入队列尾部，从首部移除、以及toString函数
TEST(TEST_DUALQUEUE, DUALQUEUE_instantiate_append_remove_toString){
    //given
    DUALQUEUE dq(5);
    dq << 1 << 2 << 3 << 4 << 5 << 6;
    int i;

    //when
    //得到对象queue的字符串表示
    std::string content = dq.toString();

    //then
    ASSERT_EQ(std::string("1 2 3 4 5 6"),dq.toString());


    //when
    dq >> i;
    //then1
    ASSERT_EQ(1,i);
    dq >> i;
    //then
    ASSERT_EQ(2,i);
     dq >> i;
    //then
    ASSERT_EQ(3,i);
    //then
    ASSERT_EQ(std::string("4 5 6"),dq.toString());

    //移除所有元素
    //
    dq >> i >> i >> i ;
    //then
    ASSERT_EQ(6,i);
    //then
    //得到对象queue的字符串表示
    ASSERT_EQ(std::string(""),dq.toString());
}