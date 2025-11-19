// ChatRoomClientGUIDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "ChatRoomClientGUI.h"
#include "ChatRoomClientGUIDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()


// CChatRoomClientGUIDlg 对话框




CChatRoomClientGUIDlg::CChatRoomClientGUIDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CChatRoomClientGUIDlg::IDD, pParent)
	, m_Port(5050)
	, m_SendMsg("")
	, m_Socket(NULL)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_Socket = new CSessionSocket(); //创建会话socket对象
}

CChatRoomClientGUIDlg::~CChatRoomClientGUIDlg(){
	if(this->m_Socket != NULL){
		this->m_Socket->Close();
		delete this->m_Socket;
	}
}

void CChatRoomClientGUIDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_IPADDRESS, m_IPCtrl);
	DDX_Text(pDX, IDC_PORT, m_Port);
	DDX_Control(pDX, IDC_MSGLIST, m_List);
	DDX_Text(pDX, IDC_SENDMSG, m_SendMsg);
}

BEGIN_MESSAGE_MAP(CChatRoomClientGUIDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_SEND, &CChatRoomClientGUIDlg::OnBnClickedSend)
	ON_BN_CLICKED(IDC_CONN, &CChatRoomClientGUIDlg::OnBnClickedConn)
	ON_BN_CLICKED(IDC_QUIT, &CChatRoomClientGUIDlg::OnBnClickedQuit)
END_MESSAGE_MAP()


// CChatRoomClientGUIDlg 消息处理程序

BOOL CChatRoomClientGUIDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// 创建套接字
	this->m_Socket->Create(0);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CChatRoomClientGUIDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CChatRoomClientGUIDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CChatRoomClientGUIDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



// 发送用户输入的消息
void CChatRoomClientGUIDlg::OnBnClickedSend()
{
	UpdateData(TRUE); //得到用户在输入编辑框里输入的字符串

	this->m_Socket->Send(this->m_SendMsg,this->m_SendMsg.GetLength());

	//清空编辑框
	this->m_SendMsg = "";
	UpdateData(FALSE);
}

//连接服务器
void CChatRoomClientGUIDlg::OnBnClickedConn()
{
	//得到用户输入的IP地址
	BYTE f0,f1,f2,f3;
	CIPAddressCtrl *ipCtrl = (CIPAddressCtrl *)this->GetDlgItem(IDC_IPADDRESS);
	ipCtrl->GetAddress(f0,f1,f2,f3);
	CString ip;
	ip.Format(_T("%d.%d.%d.%d"),f0,f1,f2,f3);

	//得到用户输入的端口号
	UpdateData(TRUE);

	//与服务器建立连接
	this->m_Socket->Connect(ip,this->m_Port);
	(CButton *)(this->GetDlgItem(IDC_SEND))->EnableWindow(TRUE);
}

void CChatRoomClientGUIDlg::OnOK()
{
	// TODO: Add your specialized code here and/or call the base class

	//这里什么都不做，防止用户按Enter键关闭窗口
}

void CChatRoomClientGUIDlg::OnCancel()
{
	// TODO: Add your specialized code here and/or call the base class
	//这里什么都不做，防止用户按Esc键关闭窗口

}

void CChatRoomClientGUIDlg::OnBnClickedQuit()
{
	this->EndDialog(TRUE);//关闭对话框
}

//重载PreTranslateMessage，使得 消息编辑框能处理回车键，当用户输入内容按回车键客发送消息
BOOL CChatRoomClientGUIDlg::PreTranslateMessage(MSG* pMsg)
{
	if(pMsg->wParam == VK_RETURN){
		HWND hWnd = ::GetFocus();
		if(IDC_SENDMSG == ::GetDlgCtrlID(hWnd)){
			this->OnBnClickedSend();
		}
	}

	return CDialog::PreTranslateMessage(pMsg);
}
