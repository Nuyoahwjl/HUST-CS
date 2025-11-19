#pragma once
#include "RdtSender.h"
//#include "SeqStateEnum.h"
class SRRdtSender :
	public RdtSender
{
private:
	const int seqsize; //序号大小，实验建议二进制为3位，即为8
	const int wndsize; //滑动窗口大小，实验建议为4
	Packet* const sendBuf;//发送缓冲区，避免反复构造析构
	bool* const bufStatus;
	int base, nextSeqnum;

private:
	void Init();
	void printSlideWindow();
	bool isInWindow(int seqnum);

public:
	SRRdtSender(int sSize, int wsize);
	SRRdtSender();
	bool send(const Message& message);
	bool getWaitingState();
	void timeoutHandler(int seqnum);
	void receive(const Packet& ackPkt);
	virtual ~SRRdtSender();
};