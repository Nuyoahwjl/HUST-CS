#pragma once
#include "winsock2.h"
#include <stdio.h>
#include <iostream>
#include <string>

#pragma comment(lib,"ws2_32.lib")

using namespace std;

void main(){
	WSADATA wsaData;
	string input;

	int nRc = WSAStartup(0x0202,&wsaData);

	if(nRc){
		printf("Winsock  startup failed with error!\n");
	}

	if(wsaData.wVersion != 0x0202){
		printf("Winsock version is not correct!\n");
	}

	printf("Winsock  startup Ok!\n");


	SOCKET clientSocket;
	sockaddr_in serverAddr,clientAddr;

	int addrLen;

	//create socket
	clientSocket = socket(AF_INET,SOCK_STREAM,0);

	if(clientSocket != INVALID_SOCKET)
		printf("Socket create Ok!\n");

	//set client port and ip
	clientAddr.sin_family = AF_INET;
	clientAddr.sin_port = htons(0);
	clientAddr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
	//binding
	int rtn = bind(clientSocket,(LPSOCKADDR)&clientAddr,sizeof(clientAddr));
	if(rtn != SOCKET_ERROR)
		printf("Socket bind Ok!\n");

	//set server's ip and port
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_port = htons(5050);
	serverAddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");

	rtn = connect(clientSocket,(LPSOCKADDR)&serverAddr,sizeof(serverAddr));
	if(rtn == SOCKET_ERROR )
		printf("Connect to server error!\n");

	printf("Connect to server ok!");

	do {
		cout << "\nPlease input your message:";
		cin >> input;

		//send data to server
		rtn = send(clientSocket,input.c_str(),input.length(),0);
		if (rtn == SOCKET_ERROR) {
			printf("Send to server failed\n");
			closesocket(clientSocket);
			WSACleanup();
			return ;
		}
	}while(input != "quit");
	closesocket(clientSocket);
	WSACleanup();
}