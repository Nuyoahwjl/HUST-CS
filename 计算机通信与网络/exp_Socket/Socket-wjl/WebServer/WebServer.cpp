#define _WINSOCK_DEPRECATED_NO_WARNINGS
#pragma once
#include "winsock2.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <time.h>

#pragma comment(lib, "ws2_32.lib")

using namespace std;

// 服务器配置结构体
struct ServerConfig
{
	string address;
	int port;
	string root;
};

// HTTP响应状态码映射
map<int, string> statusMessages = {
	{200, "OK"},
	{400, "Bad Request"},
	{404, "Not Found"},
	{500, "Internal Server Error"}};

// MIME类型映射
map<string, string> mimeTypes = {
	{".html", "text/html"},
	{".htm", "text/html"},
	{".txt", "text/plain"},
	{".css", "text/css"},
	{".js", "application/javascript"},
	{".jpg", "image/jpeg"},
	{".jpeg", "image/jpeg"},
	{".png", "image/png"},
	{".gif", "image/gif"},
	{".bmp", "image/bmp"},
	{".ico", "image/x-icon"}};

// 读取配置文件
bool readConfig(const string &filename, ServerConfig &config)
{
	ifstream file(filename);
	if (!file.is_open())
	{
		cout << "错误: 无法打开配置文件 " << filename << endl;
		return false;
	}

	string line;
	while (getline(file, line))
	{
		// 跳过注释和空行
		if (line.empty() || line[0] == '#')
			continue;

		size_t pos = line.find('=');
		if (pos != string::npos)
		{
			string key = line.substr(0, pos);
			string value = line.substr(pos + 1);

			// 去除前后空格
			key.erase(0, key.find_first_not_of(" \t"));
			key.erase(key.find_last_not_of(" \t") + 1);
			value.erase(0, value.find_first_not_of(" \t"));
			value.erase(value.find_last_not_of(" \t") + 1);

			if (key == "address")
			{
				config.address = value;
			}
			else if (key == "port")
			{
				config.port = stoi(value);
			}
			else if (key == "root")
			{
				config.root = value;
				// 确保根目录以斜杠结尾
				if (!config.root.empty() && config.root.back() != '\\' && config.root.back() != '/')
				{
					config.root += "/";
				}
			}
		}
	}

	file.close();
	return true;
}

// 获取文件扩展名
string getFileExtension(const string &filename)
{
	size_t pos = filename.find_last_of('.');
	if (pos != string::npos)
	{
		return filename.substr(pos);
	}
	return "";
}

// 获取MIME类型
string getMimeType(const string &filename)
{
	string ext = getFileExtension(filename);
	auto it = mimeTypes.find(ext);
	if (it != mimeTypes.end())
	{
		return it->second;
	}
	return "application/octet-stream";
}

// 读取文件内容
bool readFile(const string &filepath, string &content)
{
	ifstream file(filepath, ios::binary);
	if (!file.is_open())
	{
		return false;
	}

	// 获取文件大小
	file.seekg(0, ios::end);
	streamsize size = file.tellg();
	file.seekg(0, ios::beg);

	// 读取文件内容
	content.resize(size);
	file.read(&content[0], size);
	file.close();

	return true;
}

// 构建HTTP响应头
string buildResponseHeader(int statusCode, const string &contentType, size_t contentLength)
{
	stringstream header;
	header << "HTTP/1.1 " << statusCode << " " << statusMessages[statusCode] << "\r\n";
	header << "Server: SimpleWebServer/1.0\r\n";
	header << "Content-Type: " << contentType << "\r\n";
	header << "Content-Length: " << contentLength << "\r\n";
	header << "Connection: close\r\n";
	header << "\r\n";

	return header.str();
}

// 构建错误页面
string buildErrorPage(int statusCode, const string &message)
{
	stringstream html;
	html << "<!DOCTYPE html>\n";
	html << "<html>\n";
	html << "<head>\n";
	html << "<title>Error " << statusCode << "</title>\n";
	html << "</head>\n";
	html << "<body>\n";
	html << "<h1>错误 " << statusCode << ": " << statusMessages[statusCode] << "</h1>\n";
	html << "<p>" << message << "</p>\n";
	html << "<hr>\n";
	html << "<p><em>WebServer/1.0</em></p>\n";
	html << "</body>\n";
	html << "</html>\n";

	return html.str();
}

// 解析HTTP请求
bool parseHttpRequest(const string &request, string &method, string &path, string &version)
{
	istringstream iss(request);
	iss >> method >> path >> version;

	if (method.empty() || path.empty() || version.empty())
	{
		return false;
	}

	return true;
}

// 获取当前时间字符串
string getCurrentTime()
{
	time_t now = time(0);
	char buf[100];
	ctime_s(buf, sizeof(buf), &now);
	string timeStr(buf);
	// 去除换行符
	if (!timeStr.empty() && timeStr.back() == '\n')
	{
		timeStr.pop_back();
	}
	return timeStr;
}

// 处理客户端请求
void handleClient(SOCKET clientSocket, sockaddr_in clientAddr, const ServerConfig &config)
{
	char buffer[4096];
	int bytesReceived;

	// 接收HTTP请求
	bytesReceived = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
	if (bytesReceived <= 0)
	{
		cout << "错误: 接收请求失败\n"
			 << endl;
		closesocket(clientSocket);
		return;
	}

	buffer[bytesReceived] = '\0';
	string request(buffer);

	// 解析请求行
	string method, path, version;
	if (!parseHttpRequest(request, method, path, version))
	{
		cout << "错误: 解析HTTP请求失败\n"
			 << endl;

		// 返回400错误
		string errorPage = buildErrorPage(400, "无效的HTTP请求");
		string header = buildResponseHeader(400, "text/html", errorPage.length());
		string response = header + errorPage;
		send(clientSocket, response.c_str(), response.length(), 0);

		closesocket(clientSocket);
		return;
	}

	// 输出请求信息
	cout << "[" << getCurrentTime() << "] ";
	cout << "客户端 " << inet_ntoa(clientAddr.sin_addr) << ":" << ntohs(clientAddr.sin_port);
	cout << " 请求: " << method << " " << path << " " << version << endl;

	// 只支持GET方法
	if (method != "GET")
	{
		string errorPage = buildErrorPage(400, "只支持GET方法");
		string header = buildResponseHeader(400, "text/html", errorPage.length());
		string response = header + errorPage;
		send(clientSocket, response.c_str(), response.length(), 0);

		cout << "处理结果: 400 - 方法不允许\n"
			 << endl;
		closesocket(clientSocket);
		return;
	}

	// 处理路径（默认首页）
	if (path == "/")
	{
		path = "/index.html";
	}

	// 构造实际文件路径
	string filepath = config.root + path.substr(1); // 去掉开头的'/'

	// 读取文件
	string fileContent;
	if (!readFile(filepath, fileContent))
	{
		// 文件不存在
		string errorPage = buildErrorPage(404, "文件未找到: " + path);
		string header = buildResponseHeader(404, "text/html", errorPage.length());
		string response = header + errorPage;
		send(clientSocket, response.c_str(), response.length(), 0);

		cout << "处理结果: 404 - 文件未找到\n"
			 << endl;
	}
	else
	{
		// 文件存在，发送响应
		string mimeType = getMimeType(filepath);
		string header = buildResponseHeader(200, mimeType, fileContent.length());

		// 发送响应头
		send(clientSocket, header.c_str(), header.length(), 0);
		// 发送文件内容
		send(clientSocket, fileContent.c_str(), fileContent.length(), 0);

		cout << "处理结果: 200 - 成功发送文件: " << filepath << " (" << fileContent.length() << " 字节)\n"
			 << endl;
	}

	closesocket(clientSocket);
}

int main()
{
	ServerConfig config;

	// 读取配置文件
	if (!readConfig("server.conf", config))
	{
		return 1;
	}

	cout << "==== Web服务器 ====" << endl;
	cout << "监听地址: " << config.address << endl;
	cout << "监听端口: " << config.port << endl;
	cout << "根目录: " << config.root << endl;
	cout << "===================" << endl;

	// 初始化Winsock
	WSADATA wsaData;
	int nRc = WSAStartup(0x0202, &wsaData);

	if (nRc)
	{
		cout << "错误: Winsock初始化失败!" << endl;
		return 1;
	}

	if (wsaData.wVersion != 0x0202)
	{
		cout << "错误: Winsock版本不正确!" << endl;
		WSACleanup();
		return 1;
	}

	cout << "Winsock初始化成功!" << endl;

	// 创建服务器Socket
	SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0);
	if (serverSocket == INVALID_SOCKET)
	{
		cout << "错误: 创建Socket失败!" << endl;
		WSACleanup();
		return 1;
	}

	cout << "服务器Socket创建成功!" << endl;

	// 设置服务器地址
	sockaddr_in serverAddr;
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_port = htons(config.port);

	if (config.address == "0.0.0.0")
	{
		serverAddr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
	}
	else
	{
		serverAddr.sin_addr.S_un.S_addr = inet_addr(config.address.c_str());
	}

	// 绑定Socket
	if (bind(serverSocket, (sockaddr *)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR)
	{
		cout << "错误: 绑定Socket失败!" << endl;
		closesocket(serverSocket);
		WSACleanup();
		return 1;
	}

	cout << "Socket绑定成功!" << endl;

	// 开始监听
	if (listen(serverSocket, 10) == SOCKET_ERROR)
	{
		cout << "错误: 监听失败!" << endl;
		closesocket(serverSocket);
		WSACleanup();
		return 1;
	}

	cout << "开始监听连接..." << endl;
	cout << "服务器已启动，等待客户端连接...\n"
		 << endl;

	// 主循环
	while (true)
	{
		sockaddr_in clientAddr;
		int clientAddrLen = sizeof(clientAddr);

		// 接受客户端连接
		SOCKET clientSocket = accept(serverSocket, (sockaddr *)&clientAddr, &clientAddrLen);
		if (clientSocket == INVALID_SOCKET)
		{
			cout << "错误: 接受客户端连接失败!" << endl;
			continue;
		}

		// 处理客户端请求
		handleClient(clientSocket, clientAddr, config);
	}

	// 清理资源
	closesocket(serverSocket);
	WSACleanup();

	return 0;
}