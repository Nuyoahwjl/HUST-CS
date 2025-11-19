#include "stdafx.h"
#include "GBNRdtReceiver.h"
#include "Global.h"

GBNRdtReceiver::GBNRdtReceiver() :
	seqsize(8)
{
	Init();
}

void GBNRdtReceiver::Init()
{
	expectSequenceNumberRcvd = 0;
	lastAckPkt.acknum = -1; //初始状态下，上次发送的确认包的确认序号为0，使得当第一个接受的数据包出错时该确认报文的确认号为0
	lastAckPkt.checksum = 0;
	lastAckPkt.seqnum = -1;	//忽略该字段
	for (int i = 0; i < Configuration::PAYLOAD_SIZE; i++) {
		lastAckPkt.payload[i] = '.';
	}
	lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
}

GBNRdtReceiver::GBNRdtReceiver(int sSize) :
	seqsize(sSize)
{
	Init();
}


GBNRdtReceiver::~GBNRdtReceiver()
{
}

void GBNRdtReceiver::receive(const Packet& packet)
{
	//检查校验和是否正确
	int checkSum = pUtils->calculateCheckSum(packet);

	//如果校验和正确，同时收到报文的序号等于接收方期待收到的报文序号一致
	if (checkSum == packet.checksum && this->expectSequenceNumberRcvd == packet.seqnum)
	{
		pUtils->printPacket("【接收方】正确收到发送方的报文", packet);

		//取出Message，向上递交给应用层
		Message msg;
		memcpy(msg.data, packet.payload, sizeof(packet.payload));
		pns->delivertoAppLayer(RECEIVER, msg);

		lastAckPkt.acknum = packet.seqnum; //确认序号等于收到的报文序号
		lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
		pUtils->printPacket("【接收方】发送确认报文", lastAckPkt);
		pns->sendToNetworkLayer(SENDER, lastAckPkt);	//调用模拟网络环境的sendToNetworkLayer，通过网络层发送确认报文到对方

		expectSequenceNumberRcvd = (1 + expectSequenceNumberRcvd) % seqsize; //接收序号在seqsize内递增

		std::cout << "【接收方】期待接收的下一个序号: " << expectSequenceNumberRcvd << "\n\n";
	}
	else
	{
		if (checkSum != packet.checksum) {
			pUtils->printPacket("【接收方】收到损坏的报文，数据校验错误", packet);
		}
		else {
			// 使用字符数组构建消息
			char msg[100];
			sprintf_s(msg, sizeof(msg), "【接收方】收到乱序报文，期待序号=%d但收到序号=%d", expectSequenceNumberRcvd, packet.seqnum);
			pUtils->printPacket(msg, packet);
		}
		pUtils->printPacket("【接收方】重新发送上次的确认报文", lastAckPkt);
		pns->sendToNetworkLayer(SENDER, lastAckPkt);	//调用模拟网络环境的sendToNetworkLayer，通过网络层发送上次的确认报文
		std::cout << std::endl;
	}
}