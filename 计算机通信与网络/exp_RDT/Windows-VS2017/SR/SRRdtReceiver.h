#pragma once
#include "RdtReceiver.h"
class SRRdtReceiver :
	public RdtReceiver
{
private:
	//sr协议接收方也需要缓冲区和滑动窗口
	const int wndsize;
	const int seqsize;
	Packet lastAckPkt;
	Message* const recvBuf;
	bool* const bufStatus;
	int base;//int nextSeqnum;	

private:
	void Init();
	void printSlideWindow();
	bool isInWindow(int seqnum);

public:
	SRRdtReceiver();
	SRRdtReceiver(int sSize, int wsize);
	void receive(const Packet& packet);
	virtual ~SRRdtReceiver();
};