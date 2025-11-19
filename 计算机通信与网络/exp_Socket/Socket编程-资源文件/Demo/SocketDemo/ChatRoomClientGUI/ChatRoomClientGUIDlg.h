// ChatRoomClientGUIDlg.h : 头文件
//

#pragma once
#include "afxcmn.h"
#include "afxwin.h"
#include "SessionSocket.h"


// CChatRoomClientGUIDlg 对话框
class CChatRoomClientGUIDlg : public CDialog
{
// 构造
public:
	CChatRoomClientGUIDlg(CWnd* pParent = NULL);	// 标准构造函数
	virtual ~CChatRoomClientGUIDlg();

// 对话框数据
	enum { IDD = IDD_CHATROOMCLIENTGUI_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
private:
	CIPAddressCtrl m_IPCtrl;
	UINT m_Port;
	CString m_SendMsg;
	CSessionSocket *m_Socket;
public:
	CListBox m_List;
	afx_msg void OnBnClickedSend();
	afx_msg void OnBnClickedConn();
protected:
	virtual void OnOK();
	virtual void OnCancel();
public:
	afx_msg void OnBnClickedQuit();
	virtual BOOL PreTranslateMessage(MSG* pMsg);
};
