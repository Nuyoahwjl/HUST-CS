#pragma once
#include "RdtSender.h"
#include "DataStructure.h"
class GBNRdtSender : public RdtSender
{
private:
	int base; //基序号，最早的未确认分组的序号
	int nextSeqnum;	//下一个待发分组的序号
	const int wndsize; //滑动窗口大小，实验建议为4
	const int seqsize; //序号大小，实验建议二进制为3位，即为8
	Packet* const sendBuf; //发送缓冲区，保存发送的报文，用于重传

private:
	void Init();
	bool isInWindow(int seqnum);
	void printSlideWindow();

public:
	GBNRdtSender();
	GBNRdtSender(int wsize, int sSize);
	virtual ~GBNRdtSender();

	bool getWaitingState();
	bool send(const Message& message);
	void timeoutHandler(int seqNum);
	void receive(const Packet& ackPkt);
};