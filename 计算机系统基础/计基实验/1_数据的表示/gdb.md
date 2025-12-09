
### 1. 编译程序以便调试
使用 `-g` 选项编译代码，以确保 GDB 可以读取调试符号：

```bash
g++ -g your_program.cpp -o your_program
```

### 2. 启动 GDB
通过以下命令启动 GDB：

```bash
gdb ./your_program
```

### 3. 设置断点
在 GDB 中，最常见的调试操作是设置断点。可以通过以下几种方式设置断点：

- **按函数名设置断点**：
  ```bash
  break main
  ```
  这将在 `main()` 函数的起始处设置一个断点。

- **按行号设置断点**：
  ```bash
  break your_program.cpp:10
  ```
  这将在 `your_program.cpp` 文件的第 10 行设置断点。

- **按条件设置断点**：
  ```bash
  break your_program.cpp:10 if x > 5
  ```
  这将在 `your_program.cpp` 第 10 行，当变量 `x` 大于 5 时暂停。

### 4. 运行程序
启动程序运行到断点：

```gdb
run
```

程序将停止在断点处。

### 5. 查看第一个学生的 `score`
程序停在断点后，你可以查看第一个学生的 `score`（它是 `old_s[0].score`），并查看其在内存中的浮点数编码。

1. 首先，打印 `old_s[0].score`，查看其值：

   ```gdb
   print old_s[0].score
   ```

2. 如果你想查看 `score` 的内存中的编码表示，可以使用 `x` 命令查看它在内存中的字节表示。假设 `score` 是 `old_s[0]` 中的成员，执行以下命令：

   ```gdb
   x/4xb &old_s[0].score
   ```

   - `x/4xb` 命令表示查看 4 个字节，以 16 进制显示。
   - `&old_s[0].score` 是第一个学生的 `score` 的地址。

这会显示 `score` 在内存中的 4 字节表示，按照 IEEE 754 浮点数格式存储。

### 6. 查看 `message` 数组的内存信息：
   使用 `x` 命令查看 `message` 数组中存储的内存数据。例如，`message` 是一个 `char` 数组，你可以这样查看它的前 20 个字节：

   ```bash
   x/20xb message
   ```
   - `x` 表示查看内存（examine memory）。
   - `/20` 表示查看 20 个单位。
   - `b` 表示以字节（byte）的形式显示。
   - `message` 是变量的名称。

   如果想以十六进制的形式显示更多内容，可以修改单位和数量：
   ```bash
   x/40xb message
   ```


### 7. 继续调试
如果需要，你可以继续使用 `next` 或 `continue` 命令执行后续步骤。









