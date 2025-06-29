#include "storage/disk_manager.h"

#include <assert.h>    // for assert
#include <string.h>    // for memset
#include <sys/stat.h>  // for stat
#include <unistd.h>    // for lseek

#include "defs.h"

DiskManager::DiskManager() { memset(fd2pageno_, 0, MAX_FD * (sizeof(std::atomic<page_id_t>) / sizeof(char))); }

/**
 * @brief Write the contents of the specified page into disk file
 *
 */
void DiskManager::write_page(int fd, page_id_t page_no, const char *offset, int num_bytes) {
    // Todo:
    // 1.lseek()定位到文件头，通过(fd,page_no)可以定位指定页面及其在磁盘文件中的偏移量
    // 2.调用write()函数
    // 注意处理异常

    int flags = fcntl(fd, F_GETFD);  //fd是否可用
    if (flags == -1) {
        throw UnixError();
    }

    if(lseek(fd, page_no * PAGE_SIZE, SEEK_SET) == -1) {  //定位读写指针
        throw UnixError();
    }
    if(write(fd, offset, num_bytes) != num_bytes) { //写文件
        throw UnixError();
    }

}

/**
 * @brief Read the contents of the specified page into the given memory area
 */
void DiskManager::read_page(int fd, page_id_t page_no, char *offset, int num_bytes) {
    // Todo:
    // 1.lseek()定位到文件头，通过(fd,page_no)可以定位指定页面及其在磁盘文件中的偏移量
    // 2.调用read()函数
    // 注意处理异常

    int flags = fcntl(fd, F_GETFD);  //fd是否可用
    if (flags == -1) {
        throw UnixError();
    }

    if(lseek(fd, page_no * PAGE_SIZE, SEEK_SET) == -1) {  //定位读写指针
        throw UnixError();
    }
    if(read(fd, offset, num_bytes) == -1) { //读文件
        throw UnixError();
    }

}

/**
 * @brief Allocate new page (operations like create index/table)
 * For now just keep an increasing counter
 */
page_id_t DiskManager::AllocatePage(int fd) {
    // Todo:
    // 简单的自增分配策略，指定文件的页面编号加1
    page_id_t pidt = fd2pageno_[fd];  //获取fd的页面编号，fd2pageno_是以fd为索引的数组
    fd2pageno_[fd]+=1;
    return pidt;
}

/**
 * @brief Deallocate page (operations like drop index/table)
 * Need bitmap in header page for tracking pages
 * This does not actually need to do anything for now.
 */
void DiskManager::DeallocatePage(__attribute__((unused)) page_id_t page_id) {}

bool DiskManager::is_dir(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

void DiskManager::create_dir(const std::string &path) {
    // Create a subdirectory
    std::string cmd = "mkdir " + path;
    if (system(cmd.c_str()) < 0) {  // 创建一个名为path的目录
        throw UnixError();
    }
}

void DiskManager::destroy_dir(const std::string &path) {
    std::string cmd = "rm -r " + path;
    if (system(cmd.c_str()) < 0) {
        throw UnixError();
    }
}

/**
 * @brief 用于判断指定路径文件是否存在 
 */
bool DiskManager::is_file(const std::string &path) {
    // Todo:
    // 用struct stat获取文件信息
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

/**
 * @brief 用于创建指定路径文件
 */
void DiskManager::create_file(const std::string &path) {
    // Todo:
    // 调用open()函数，使用O_CREAT模式
    // 注意不能重复创建相同文件
    if(this->is_file(path)) {  //判断文件是否已经存在，调用上方函数
        throw FileExistsError(path);
    }
    int fd = open(path.c_str(), O_CREAT | O_RDWR, 0666); //path.c_str()将std::string转换为const char*，传入O_CREAT标志位可以创建文件，添加O_RDWR标志位允许读写权限，设置文件模式为0666表示任何用户都有读写权限。
    if(fd == -1) {
        throw UnixError();
    }
    close(fd);

}

/**
 * @brief 用于删除指定路径文件 
 */
void DiskManager::destroy_file(const std::string &path) {
    // Todo:
    // 调用unlink()函数
    // 注意不能删除未关闭的文件
    if(!this->is_file(path)) {  //是否存在
        throw FileNotFoundError(path);
    }
    if(this->path2fd_.count(path)) {  //是否未关闭
        throw FileNotClosedError(path);
    }
    if(unlink(path.c_str()) < 0) {
        throw UnixError();
    }
    
}

/**
 * @brief 用于打开指定路径文件
 */
int DiskManager::open_file(const std::string &path) {
    // Todo:
    // 调用open()函数，使用O_RDWR模式
    // 注意不能重复打开相同文件，并且需要更新文件打开列表
    if(this->path2fd_.count(path)) {  // 已打开则不重复打开
        throw FileNotClosedError(path);
    }
    if(!this->is_file(path)) { // path是否正确
        throw FileNotFoundError(path);
    }
    int fd = open(path.c_str(), O_RDWR);
    if(fd < 0) {
        throw UnixError();
    }
    this->path2fd_[path] = fd;  //更新映射
    this->fd2path_[fd] = path;
    return fd;
}

/**
 * @brief 用于关闭指定路径文件
 */
void DiskManager::close_file(int fd) {
    // Todo:
    // 调用close()函数
    // 注意不能关闭未打开的文件，并且需要更新文件打开列表

    if(!this->fd2path_.count(fd)) { // 未打开则不关闭
        throw FileNotOpenError(fd);
        return;
    }
    if(close(fd) == -1) {
        throw UnixError();
        return;
    }
    auto it1 = path2fd_.find(this->GetFileName(fd)); //删除path2fd中相应的映射
    if (it1 != path2fd_.end()) {
        path2fd_.erase(it1);
    }

    auto it2 = fd2path_.find(fd); //删除fd2path中相应的映射
    if (it2 != fd2path_.end()) {
        fd2path_.erase(it2);
    }
    
}

int DiskManager::GetFileSize(const std::string &file_name) {
    struct stat stat_buf;
    int rc = stat(file_name.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

std::string DiskManager::GetFileName(int fd) {
    if (!fd2path_.count(fd)) {
        throw FileNotOpenError(fd);
    }
    return fd2path_[fd];
}

int DiskManager::GetFileFd(const std::string &file_name) {
    if (!path2fd_.count(file_name)) {
        return open_file(file_name);
    }
    return path2fd_[file_name];
}

bool DiskManager::ReadLog(char *log_data, int size, int offset, int prev_log_end) {
    // read log file from the previous end
    if (log_fd_ == -1) {
        log_fd_ = open_file(LOG_FILE_NAME);
    }
    offset += prev_log_end;
    int file_size = GetFileSize(LOG_FILE_NAME);
    if (offset >= file_size) {
        return false;
    }

    size = std::min(size, file_size - offset);
    lseek(log_fd_, offset, SEEK_SET);
    ssize_t bytes_read = read(log_fd_, log_data, size);
    if (bytes_read != size) {
        throw UnixError();
    }
    return true;
}

void DiskManager::WriteLog(char *log_data, int size) {
    if (log_fd_ == -1) {
        log_fd_ = open_file(LOG_FILE_NAME);
    }

    // write from the file_end
    lseek(log_fd_, 0, SEEK_END);
    ssize_t bytes_write = write(log_fd_, log_data, size);
    if (bytes_write != size) {
        throw UnixError();
    }
}