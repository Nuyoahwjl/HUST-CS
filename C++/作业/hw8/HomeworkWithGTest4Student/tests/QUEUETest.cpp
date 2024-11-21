#include <gtest/gtest.h>

#include "../include/ch6_ch8_ch11_homework/QUEUE.h"
#include <iostream>

/*
    测试TEST_QUEUE的实例化、将元素加入队列尾部以及toString函数
*/
TEST(TEST_QUEUE, QUEUE_Test_instantiate_append_toString){
    //given
    QUEUE queue(5);
    queue << 1 << 2 << 3 << 4 << 5;

    //when
    //得到对象queue的字符串表示
    std::string content = queue.toString();

    //then
    std::string target("1 2 3 4 5");
    ASSERT_EQ(target,content);
}

/*
    测试TEST_QUEUE的实例化、将元素移除队列首部部以及toString函数
*/
TEST(TEST_QUEUE, QUEUE_Test_instantiate_remove_toString){
    //given
    QUEUE queue(5);
    queue << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素
    int i;

    //when1
    //移除队首元素
    queue >> i;

    //then1_1
    ASSERT_EQ(1,i);

    //then1_2
    ASSERT_EQ(std::string("2 3 4 5"),queue.toString());

    //when2
    //移除所有元素
    queue >> i >> i >> i >> i ;
    //then2_1
    ASSERT_EQ(5,i);

    //then2_2
    //得到对象queue的字符串表示
    ASSERT_EQ(std::string(""),queue.toString());
}

/*
    测试QUEUE的拷贝构造函数
*/
TEST(TEST_QUEUE, QUEUE_Test_CopyConstructor){
    //given
    QUEUE qa(5);
    qa << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素

    //检测内容是否相等
    //when
    QUEUE qb(qa);
    //then
    ASSERT_EQ(qa.toString(),qb.toString());

    //检查是否为深拷贝
    //when
    qa.~QUEUE();  //析构对象qa
    //then
    std::string target("1 2 3 4 5");
    ASSERT_EQ(target,qb.toString()); //这时对象qb的内容应该还在，深拷贝
}

/**

    测试QUEUE的copy assignment（拷贝=）
*/
TEST(TEST_QUEUE, QUEUE_Test_CopyAssignment){
    //given
    QUEUE qa(1);
    QUEUE qb(5);
    qb << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素

    //检测内容是否相等
    //when
    qa = qb;
    //then
    ASSERT_EQ(qa.toString(),qb.toString());

    //检查是否为深拷贝
    //when
    qb.~QUEUE();
    //then
    std::string target("1 2 3 4 5");
    ASSERT_EQ(target,qa.toString()); 
}

/*
    测试QUEUE的移动构造函数
*/
TEST(TEST_QUEUE, QUEUE_Test_MoveConstructor){
    //given
    QUEUE qa(5);
    qa << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素
    const int * const pa = qa.getElemsPtr();  //保留qa的内部指针

    //when
    QUEUE qb(std::move(qa));

    //检测内容是否相等
    std::string target("1 2 3 4 5");
    //then
    ASSERT_EQ(target,qb.toString());

    //检查是否为浅拷贝
    //then
    ASSERT_EQ(pa,qb.getElemsPtr());

    //检查qa对象是不是为可以安全析构
    //顺便检查了operator int()
    //then
    ASSERT_EQ(0,qa);
    ASSERT_EQ(0,qa.capacity());
    ASSERT_EQ(nullptr,qa.getElemsPtr());
}

/*
    测试QUEUE的move assignment（移动=）
*/
TEST(TEST_QUEUE, QUEUE_Test_MoveAssignment){
    //given
    QUEUE qa(1);
    QUEUE qb(5);
    qb << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素
    const int * const pb = qb.getElemsPtr();  //保留qb的内部指针

    //when
    qa = std::move(qb);

    //检测内容是否相等
    std::string target("1 2 3 4 5");
    //then
    ASSERT_EQ(target,qa.toString());

    //检查是否为浅拷贝
    //then
    ASSERT_EQ(pb,qa.getElemsPtr());

    //检查qb对象是不是为可以安全析构
    //顺便检查了operator int()
    //then
    ASSERT_EQ(0,qb);
    ASSERT_EQ(0,qb.capacity());
    ASSERT_EQ(nullptr,qb.getElemsPtr());

}

/*
    测试QUEUE的析构函数
*/
TEST(TEST_QUEUE, QUEUE_Test_Deconstructor){
    //given
    QUEUE qa(5);
    qa << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素
    int rtn = 0;

    //when
    qa.~QUEUE();
    qa.~QUEUE();
    rtn = 1;    //如果析构函数实现正确，会执行到rtn = 1;

    //then
    ASSERT_EQ(0,qa);
    ASSERT_EQ(nullptr,qa.getElemsPtr());
    ASSERT_EQ(1, rtn);

}

/*
    测试QUEUE的类型转换和capacity函数
*/
TEST(TEST_QUEUE, QUEUE_Test_TypeCast_Capacity){
    //given
    QUEUE qa(5);

    //then1
    ASSERT_EQ(0,qa);
    ASSERT_EQ(5,qa.capacity());

    //when
    qa << 1 << 2 << 3 << 4 << 5;  //队列加入五个元素

    //then2
    ASSERT_EQ(5,qa);
    ASSERT_EQ(5,qa.capacity());
}

