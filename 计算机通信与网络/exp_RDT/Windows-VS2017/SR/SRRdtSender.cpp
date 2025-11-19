#include "stdafx.h"
#include "SRRdtSender.h"
#include "Tool.h"
#include "Global.h"

void SRRdtSender::Init()
{
	base = nextSeqnum = 0;
	for (int i = 0; i < seqsize; i++)
		bufStatus[i] = false;
}

void SRRdtSender::printSlideWindow()
{
	std::cout << "[";
	for (int i = 0; i < seqsize; i++)
	{
		if (i == base)
			std::cout << " [";

		if (isInWindow(i)) {
			if (i == nextSeqnum)
				std::cout << " " << i << ">";  // 下一个要发送的序号用>标记
			else if (bufStatus[i] == true)
				std::cout << " " << i << "*";  // 已确认的序号用*标记
			else
				std::cout << " " << i;         // 已发送但未确认的序号
		}
		else {
			std::cout << " -";  // 不在窗口内的用-表示
		}

		if (i == (base + wndsize - 1) % seqsize)
			std::cout << "] ";
	}
	std::cout << "]";

	// 额外显示窗口信息
	std::cout << " (base=" << base << ", nextSeq=" << nextSeqnum << ", 窗口大小=" << wndsize << ")";
}

//判断序列号是否在窗口内
bool SRRdtSender::isInWindow(int seqnum)
{
	if (base < (base + wndsize) % seqsize)
		return seqnum >= base && seqnum < (base + wndsize) % seqsize;
	else
		return seqnum >= base || seqnum < (base + wndsize) % seqsize;
}

SRRdtSender::SRRdtSender(int sSize, int wsize) :
	seqsize(sSize), wndsize(wsize), sendBuf(new Packet[sSize]), bufStatus(new bool[sSize])
{
	Init();
}

SRRdtSender::SRRdtSender() :
	seqsize(8), wndsize(4), sendBuf(new Packet[8]), bufStatus(new bool[8])
{
	Init();
}

bool SRRdtSender::send(const Message& message)
{
	if (getWaitingState())
	{//窗口已满，无法发送
		std::cout << "【发送方】窗口已满，无法发送数据\n\n";
		return false;
	}
	bufStatus[nextSeqnum] = false;
	sendBuf[nextSeqnum].acknum = -1;
	sendBuf[nextSeqnum].seqnum = nextSeqnum;
	memcpy(sendBuf[nextSeqnum].payload, message.data, sizeof(message.data));
	sendBuf[nextSeqnum].checksum = pUtils->calculateCheckSum(sendBuf[nextSeqnum]);
	pUtils->printPacket("【发送方】发送数据包", sendBuf[nextSeqnum]);
	//发送分组
	pns->sendToNetworkLayer(RECEIVER, sendBuf[nextSeqnum]);
	//启动计时器，SR协议中每个分组都有独立的计时器
	pns->startTimer(SENDER, Configuration::TIME_OUT, nextSeqnum);
	//发送完毕，更新状态
	nextSeqnum = (nextSeqnum + 1) % seqsize;
	std::cout << "【发送方】发送后窗口状态：";
	printSlideWindow();
	std::cout << std::endl;
	return true;
}


bool SRRdtSender::getWaitingState()
{
	return (base + wndsize) % seqsize == (nextSeqnum) % seqsize;
}

void SRRdtSender::receive(const Packet& ackPkt)
{
	int checksum = pUtils->calculateCheckSum(ackPkt);
	if (checksum != ackPkt.checksum)
	{
		pUtils->printPacket("【发送方】收到损坏的确认包", ackPkt);
		return;
	}
	else
	{
		pns->stopTimer(SENDER, ackPkt.acknum);
		if (isInWindow(ackPkt.acknum))
		{
			//更新窗口状态
			bufStatus[ackPkt.acknum] = true;
			while (bufStatus[base] == true)
			{
				//移动base
				bufStatus[base] = false;
				base = (base + 1) % seqsize;
			}
			pUtils->printPacket("【发送方】收到有效确认包", ackPkt);
			std::cout << "【发送方】收到确认后窗口状态：";
			printSlideWindow();
			std::cout << std::endl;
		}
		else
		{
			pUtils->printPacket("【发送方】收到窗口外分组的确认包", ackPkt);
		}
	}
}

void SRRdtSender::timeoutHandler(int seqnum)
{
	std::cout << "【发送方】分组 " << seqnum << " 发生超时，开始重传...\n";
	pUtils->printPacket("【发送方】超时重传数据包", sendBuf[seqnum]);
	pns->sendToNetworkLayer(RECEIVER, sendBuf[seqnum]);
	pns->stopTimer(SENDER, seqnum);
	pns->startTimer(SENDER, Configuration::TIME_OUT, seqnum);
}

SRRdtSender::~SRRdtSender()
{
	delete[] bufStatus;
	delete[] sendBuf;
}