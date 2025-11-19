#include "stdafx.h"
#include "SRRdtReceiver.h"
#include "Global.h"

void SRRdtReceiver::Init()
{
	base = 0;
	for (int i = 0; i < seqsize; i++)
		bufStatus[i] = false;
	lastAckPkt.acknum = -1;
	lastAckPkt.checksum = 0;
	lastAckPkt.seqnum = -1;	//忽略该字段
	memset(lastAckPkt.payload, '.', Configuration::PAYLOAD_SIZE);
	lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
}

void SRRdtReceiver::printSlideWindow()
{
	std::cout << "[";
	for (int i = 0; i < seqsize; i++)
	{
		if (i == base)
			std::cout << " [";

		if (isInWindow(i)) {
			if (bufStatus[i] == true)
				std::cout << " " << i << "*";  // 已缓存的序号用*标记
			else
				std::cout << " " << i;         // 窗口内但未缓存的序号
		}
		else {
			std::cout << " -";  // 不在窗口内的用-表示
		}

		if (i == (base + wndsize - 1) % seqsize)
			std::cout << "] ";
	}
	std::cout << "]";

	// 额外显示窗口信息
	std::cout << " (base=" << base << ", 窗口大小=" << wndsize << ")";
}

bool SRRdtReceiver::isInWindow(int seqnum)
{
	if (base < (base + wndsize) % seqsize)
		return seqnum >= base && seqnum < (base + wndsize) % seqsize;
	else
		return seqnum >= base || seqnum < (base + wndsize) % seqsize;
}

SRRdtReceiver::SRRdtReceiver() :
	seqsize(8), wndsize(4), recvBuf(new Message[seqsize]), bufStatus(new bool[seqsize])
{
	Init();
}

SRRdtReceiver::SRRdtReceiver(int sSize, int wsize) :
	seqsize(sSize), wndsize(wsize), recvBuf(new Message[seqsize]), bufStatus(new bool[seqsize])
{
	Init();
}

void SRRdtReceiver::receive(const Packet& packet)
{
	int checksum = pUtils->calculateCheckSum(packet);
	if (checksum != packet.checksum)
	{
		//数据包损坏，不作出应答
		pUtils->printPacket("【接收方】收到损坏的数据包，校验和错误", packet);
		return;
	}
	else
	{
		if (isInWindow(packet.seqnum) == false)
		{
			//不是窗口内的分组，发送否定确认
			pUtils->printPacket("【接收方】收到窗口外的分组，发送否定确认", packet);
			lastAckPkt.acknum = packet.seqnum;
			lastAckPkt.seqnum = -1;
			memset(lastAckPkt.payload, '.', Configuration::PAYLOAD_SIZE);
			lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
			pns->sendToNetworkLayer(SENDER, lastAckPkt);
			return;
		}
		else
		{
			//是窗口内的分组，发送ack，更新缓冲区和滑动窗口
			if (packet.seqnum == base)
			{
				Message msg;
				memcpy(msg.data, packet.payload, sizeof(packet.payload));
				int flag = base;
				//查找连续接收的最大序号
				for (int i = (base + 1) % seqsize, j = 1; j < wndsize; j++, i = (i + 1) % seqsize)
				{
					if (bufStatus[i] == true)
						flag = i;
					else
						break;
				}
				if (flag == base)
				{
					//只有base被接收，直接交付
					pns->delivertoAppLayer(RECEIVER, msg);
				}
				else
				{
					//有连续接收的分组，一起交付
					pns->delivertoAppLayer(RECEIVER, msg);
					for (int i = (base + 1) % seqsize, j = 0; j < (flag - base + seqsize) % seqsize; j++, i = (i + 1) % seqsize) {
						pns->delivertoAppLayer(RECEIVER, recvBuf[i]);
						bufStatus[i] = false;
					}
				}
				base = (flag + 1) % seqsize;
				std::cout << "【接收方】窗口移动后状态：";
				printSlideWindow();
				std::cout << std::endl;
			}
			else
			{
				memcpy(recvBuf[packet.seqnum].data, packet.payload, sizeof(packet.payload));
				bufStatus[packet.seqnum] = true;
				std::cout << "【接收方】报文序号中断，缓存报文序号 " << packet.seqnum << "，base=" << base << std::endl;
			}

			lastAckPkt.acknum = packet.seqnum; //确认序号等于收到的报文序号
			lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
			pUtils->printPacket("【接收方】发送确认报文", lastAckPkt);
			pns->sendToNetworkLayer(SENDER, lastAckPkt);	//调用模拟网络环境的sendToNetworkLayer，通过网络层发送确认报文到对方
		}
	}

}


SRRdtReceiver::~SRRdtReceiver()
{
	delete[] recvBuf;
	delete[] bufStatus;
}