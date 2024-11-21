#pragma once
#include <string>

/**
 * 先进先出的队列
 */
class QUEUE
{
	int   max;	        //elems申请的最大元素个数为max
    int   size;         //队列里实际元素的个数
	int*  elems;	    //elems申请内存用于存放队列的元素

public:
	QUEUE(int m);									//初始化队列：队列最多存放m个元素
	QUEUE(const QUEUE& q); 							//用q深拷贝初始化队列
	QUEUE(QUEUE&& q)noexcept;						//用q移动初始化队列
	virtual operator int() const noexcept;			//返回队列的实际元素个数
	virtual int capacity() const noexcept;			//返回队列申请的最大元素个数max
	virtual QUEUE& operator<<(int e);  				//将e入队列尾部，并返回当前队列,如果队列满，抛出字符串类型异常："QUEUE is full!"
	virtual QUEUE& operator>>(int& e); 				//移除队首出元素到e，并返回当前队列，如果队列空，抛出字符串类型异常："QUEUE is empty!"
	virtual QUEUE& operator=(const QUEUE& q);		//深拷贝赋值并返回被赋值队列
	virtual QUEUE& operator=(QUEUE&& q)noexcept;	//移动赋值并返回被赋值队列
	virtual std::string toString();                 //队列内容变成字符串。要求元素之间用空格分开，形如这样的格式"0 1 2 3 4 5"，如果队列为空则返回"
	virtual ~QUEUE();	 							//销毁当前队列

	//返回内部p指针，仅仅用于测试，不能用于任何其他地方
	const int* const getElemsPtr() { return elems;}

	
};