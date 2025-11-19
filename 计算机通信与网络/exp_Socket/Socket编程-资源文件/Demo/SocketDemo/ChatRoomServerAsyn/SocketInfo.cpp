#pragma once
#include "SocketInfo.h"
#include <list>


SocketInfo::SocketInfo(SOCKET socket)
{	
	this->socket = socket;
	//this->sendString.empty();
}

SocketInfo::~SocketInfo(void)
{
	//for(list<char *>)
}
