#pragma once
#include <winsock2.h>
#include <list>

using namespace std;

class SocketInfo
{
private:
	SOCKET socket;
	list<char *> recvStrings;
	list<char *> sendStrings;
public:
	SocketInfo(SOCKET socket);
	~SocketInfo(void);

};
