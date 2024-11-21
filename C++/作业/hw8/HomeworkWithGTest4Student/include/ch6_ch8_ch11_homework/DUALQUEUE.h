#pragma once
#include "QUEUE.h"
/**
 * 在二个先进先出QUEUE上实现双队列
 * 注意DUALQUEUE内部有二个QUEUE，一个是继承的QUEUE（为了方便描述，称为入队列），一个是数据成员q的QUEUE（为了方便描述，称为出队列）
 * 当向DUALQUEUE里放入元素时，一律放到入队列；如果入队列满了，将入队列元素移动到出队列，将入列队清空；后面入栈元素继续放在入队列
 * 从DUALQUEUE取元素时，一律从出队列取元素，如果出队列为空了，要将将入队列元素移动到出队列，继续从出队列取元素
 * 
 */
class DUALQUEUE :public QUEUE
{
	QUEUE q;
public:
	DUALQUEUE(int m);                    		        	//初始化DUALQUEUE：入队列和出队列2大小都为m。DUALQUEUE最多存放2m个元素
	DUALQUEUE(const DUALQUEUE& s);         			        //深拷贝构造
	DUALQUEUE(DUALQUEUE&& s)noexcept;     			        //浅拷贝构造
	using QUEUE::operator=;                              // 引入基类的赋值运算符
	virtual int  capacity()const noexcept;		  			//返回DUALQUEUE的容量
	virtual operator int() const noexcept;	   				//返回DUALQUEUE的实际元素个数
	virtual DUALQUEUE& operator<<(int e); 	     			//将e放入DUALQUEUE，并返回当前DUALQUEUE。如果DUALQUEUE满，抛出字符串类型异常："DUALQUEUE is full!"
	virtual DUALQUEUE& operator>>(int& e);     				//从DUALQUEUE取元素到e，并返回当前DUALQUEUE。如果DUALQUEUE空，抛出字符串类型异常："DUALQUEUE is empty!"
	virtual DUALQUEUE& operator=(const DUALQUEUE& s);		//深拷贝赋值
	virtual DUALQUEUE& operator=(DUALQUEUE&& s)noexcept;	//移动赋值
	virtual std::string toString();                 		//DUALQUEUE内容变成字符串。要求元素之间用空格分开，形如这样的格式"0 1 2 3 4 5".如果DUALQUEUE为空则返回"
	~DUALQUEUE()noexcept;	              		        	//销毁DUALQUEUE

	//返回内部入队列p指针，仅仅用于测试，不能用于任何其他地方
	const int* const getInQueueElemsPtr() { return QUEUE::getElemsPtr();}
	//返回内部出队列p指针，仅仅用于测试，不能用于任何其他地方
	const int* const getOutQueueElemsPtr() { return q.getElemsPtr();}
};