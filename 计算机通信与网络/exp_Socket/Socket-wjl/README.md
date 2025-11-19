### 配置服务器

1. 编辑 `server.conf` 文件：
```ini
# Web服务器配置文件
address=0.0.0.0    # 监听地址 (0.0.0.0表示主机上任意一块网卡的IP地址)
port=8080          # 监听端口（1-65535）
root=./content     # 网站根目录路径
```

2. 准备网站文件：
   - 将文件放置在 `content` 目录下
   - 确保存在 `index.html` 作为默认首页

### 运行服务器

```bash
WebServer.exe
```

服务器启动后将显示：
```
=== 简单Web服务器 ===
监听地址: 0.0.0.0
监听端口: 8080
根目录: ./content
====================
Winsock初始化成功!
服务器Socket创建成功!
Socket绑定成功!
开始监听连接...
服务器已启动，等待客户端连接...
```

### 访问测试

在浏览器中访问：
```
http://localhost:8080/
http://localhost:8080/index.html
http://localhost:8080/HUST.png
http://localhost:8080/coding.gif
http://localhost:8080/style.css
http://localhost:8080/post.html
......
```

### 支持的MIME类型

| 文件扩展名 | MIME类型 | 说明 |
|-----------|----------|------|
| .html, .htm | text/html | HTML文档 |
| .txt | text/plain | 纯文本文件 |
| .css | text/css | 样式表 |
| .js | application/javascript | JavaScript文件 |
| .jpg, .jpeg | image/jpeg | JPEG图像 |
| .png | image/png | PNG图像 |
| .gif | image/gif | GIF图像 |
| .bmp | image/bmp | BMP图像 |
| .ico | image/x-icon | 网站图标 |

### 访问日志示例

服务器运行时会在控制台输出详细的访问日志：

```
[Thu Oct 23 21:18:42 2025] 客户端 127.0.0.1:57220 请求: GET / HTTP/1.1
处理结果: 200 - 成功发送文件: ./content/index.html (616 字节)

[Thu Oct 23 21:18:42 2025] 客户端 127.0.0.1:52918 请求: GET /HUST.png HTTP/1.1
处理结果: 200 - 成功发送文件: ./content/HUST.png (528027 字节)

[Thu Oct 23 21:18:57 2025] 客户端 127.0.0.1:56106 请求: GET /coding.gif HTTP/1.1
处理结果: 200 - 成功发送文件: ./content/coding.gif (150755 字节)

[Thu Oct 23 21:19:11 2025] 客户端 127.0.0.1:56486 请求: GET /style.css HTTP/1.1
处理结果: 200 - 成功发送文件: ./content/style.css (397 字节)

[Thu Oct 23 21:19:32 2025] 客户端 127.0.0.1:52406 请求: GET /temp.txt HTTP/1.1
处理结果: 404 - 文件未找到

[Thu Oct 23 21:19:48 2025] 客户端 127.0.0.1:55826 请求: GET /post.html HTTP/1.1
处理结果: 200 - 成功发送文件: ./content/post.html (985 字节)

[Thu Oct 23 21:19:56 2025] 客户端 127.0.0.1:65101 请求: POST /submit HTTP/1.1
处理结果: 400 - 方法不允许
```

### 错误处理

服务器能够处理以下错误情况并返回相应的HTTP状态码：

- **400 Bad Request** - 无效的HTTP请求
- **404 Not Found** - 请求的文件不存在

### 安全特性

- **请求方法验证**：仅支持安全的GET方法
- **文件存在性检查**：避免无效文件访问

### 核心组件
- **Winsock2**：Windows Socket网络通信
- **select模型**：I/O多路复用，支持非阻塞操作
- **HTTP/1.1协议**：标准HTTP协议实现
- **文件流操作**：高效的文件读取和传输

### 关键函数
- `readConfig()` - 读取配置文件
- `handleClient()` - 处理客户端请求
- `parseHttpRequest()` - 解析HTTP请求
- `buildResponseHeader()` - 构建HTTP响应头
- `readFile()` - 读取文件内容
