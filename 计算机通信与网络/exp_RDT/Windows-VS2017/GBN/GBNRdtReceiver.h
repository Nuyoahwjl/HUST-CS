#pragma once
#include "RdtReceiver.h"
#include "DataStructure.h"
class GBNRdtReceiver : public RdtReceiver
{
private:
	int expectSequenceNumberRcvd;	// 期待收到的下一个报文序号
	Packet lastAckPkt;				//上次发送的确认报文
	const int seqsize;

private:
	void Init();

public:
	GBNRdtReceiver(int sSize);
	GBNRdtReceiver();
	virtual ~GBNRdtReceiver();

public:

	void receive(const Packet& packet); //接收报文，将被NetworkService调用
};