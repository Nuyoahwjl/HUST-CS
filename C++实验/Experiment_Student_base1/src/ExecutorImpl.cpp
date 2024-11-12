#include "ExecutorImpl.hpp"
#include <new>

namespace adas
{
    // 并没有初始化ExecutorImpl的pose成员变量
    ExecutorImpl::ExecutorImpl(const Pose &pose) noexcept : pose(pose) {}

    // Query方法
    Pose ExecutorImpl::Query(void) const noexcept
    {
        return pose;
    }

    // NewExecutor方法
    Executor *Executor::NewExecutor(const Pose &pose) noexcept
    {
        return new (std::nothrow) ExecutorImpl(pose); // c++17
    }

    // Execute方法
    void ExecutorImpl::Execute(const std::string &command) noexcept
    {
        for (const auto cmd : command)
        {
            if (cmd == 'M')
            {
                if (pose.heading == 'E') { ++pose.x; }
                else if (pose.heading == 'W') { --pose.x; }
                else if (pose.heading == 'N') { ++pose.y; }
                else if (pose.heading == 'S') { --pose.y; }
            }
            else if (cmd == 'L')
            {
                if (pose.heading == 'E') { pose.heading = 'N'; }
                else if (pose.heading == 'W') { pose.heading = 'S'; }
                else if (pose.heading == 'N') { pose.heading = 'W'; }
                else if (pose.heading == 'S') { pose.heading = 'E'; }
            }
            else if (cmd == 'R')
            {
                if (pose.heading == 'E') { pose.heading = 'S'; }
                else if (pose.heading == 'W') { pose.heading = 'N'; }
                else if (pose.heading == 'N') { pose.heading = 'E'; }
                else if (pose.heading == 'S') { pose.heading = 'W'; }
            }
        }
    }
}