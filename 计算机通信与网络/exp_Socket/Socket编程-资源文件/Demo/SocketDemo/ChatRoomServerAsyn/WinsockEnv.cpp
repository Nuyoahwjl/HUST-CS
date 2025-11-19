#include "WinsockEnv.h"
#include <iostream>
#include <winsock2.h>
#pragma comment(lib, "Ws2_32.lib")

using namespace std;


#define MAJORVERSION 2  //Winsock主版本号
#define MINORVERSION 2	//Winsock次版本号

WinsockEnv::WinsockEnv(void)
{
}

WinsockEnv::~WinsockEnv(void)
{
}

int WinsockEnv::Startup(){

	WSADATA wsaData;//用于返回Winsock的环境信息   
	int rtn;		//Winsock初始化函数的返回值
	WORD  wVersionRequested = MAKEWORD( MAJORVERSION, MINORVERSION ); //Winsock版本号，高字节次版本，低字节主版本号


	//初始化 winsock
	rtn = WSAStartup(wVersionRequested, &wsaData);

	if(rtn){ //如果返回值不是0，则Winsock初始化失败
		cout<< "Winsock startup error!\n";
		return -1;
	}

	//判断返回的Winsock版本号
	if(LOBYTE(wsaData.wVersion) != MAJORVERSION || HIBYTE(wsaData.wVersion) != MINORVERSION){ //如果版本不对
		WSACleanup();  //清除Winsock
		cout << "Winsock version error!\n";
		return -1;
	}

	cout << "Winsock startup ok!\n";
	return 0;
}
