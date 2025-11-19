// SessionSocket.cpp : implementation file
//

#include "stdafx.h"
#include "ChatRoomClientGUI.h"
#include "SessionSocket.h"
#include "ChatRoomClientGUIDlg.h"


// CSessionSocket

CSessionSocket::CSessionSocket()
{
}

CSessionSocket::~CSessionSocket()
{
}






void CSessionSocket::OnReceive(int nErrorCode)
{
	char recvBuf[1024];
	int rtn;

	rtn = this->Receive(recvBuf,1024);
	switch(rtn){
		case 0:
			this->Close();
			break;
		case SOCKET_ERROR:
			if(this->GetLastError() != WSAEWOULDBLOCK){
				this->Close();
			}
			break;
		default:
			recvBuf[rtn] = '\0';
	}

	//在对话框里显示
	CChatRoomClientGUIDlg *dlg =  (CChatRoomClientGUIDlg *)(AfxGetApp()->GetMainWnd());
	dlg->m_List.AddString(CString(recvBuf));

	CAsyncSocket::OnReceive(nErrorCode);
}

void CSessionSocket::OnClose(int nErrorCode)
{
	//在对话框里显示
	CChatRoomClientGUIDlg *dlg =  (CChatRoomClientGUIDlg *)(AfxGetApp()->GetMainWnd());
	dlg->m_List.AddString("服务器关闭 !");

	CAsyncSocket::OnClose(nErrorCode);
}
