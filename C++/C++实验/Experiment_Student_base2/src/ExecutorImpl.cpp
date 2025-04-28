#include "ExecutorImpl.hpp"
#include <new>
#include <memory>

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
        // for (const auto cmd : command)
        // {
        //     if (cmd == 'M')
        //     {
        //         // Move();
        //         std::unique_ptr<MoveCommand> cmder=std::make_unique<MoveCommand>();
        //         cmder->DoOperate(*this);
        //     }
        //     else if (cmd == 'L')
        //     {
        //         // TurnLeft();
        //         std::unique_ptr<TurnLeftCommand> cmder=std::make_unique<TurnLeftCommand>();
        //         cmder->DoOperate(*this);
        //     }
        //     else if (cmd == 'R')
        //     {
        //         // TurnRight();
        //         std::unique_ptr<TurnRightCommand> cmder=std::make_unique<TurnRightCommand>();
        //         cmder->DoOperate(*this);
        //     }
        // }
        for (const auto cmd : command)
        {
            std::unique_ptr<ICommand> cmder;
            
            if (cmd == 'M')
                cmder = std::make_unique<MoveCommand>();
            else if (cmd == 'L')
                cmder = std::make_unique<TurnLeftCommand>();
            else if (cmd == 'R')
                cmder = std::make_unique<TurnRightCommand>();
            else if (cmd == 'F')
                cmder = std::make_unique<FastCommand>();

            if(cmder)
                cmder->DoOperate(*this);
        }
    }

    // Move方法
    void ExecutorImpl::Move(void) noexcept
    {
        if (pose.heading == 'E') { pose.x += 1; }
        else if (pose.heading == 'W') { pose.x -= 1; }
        else if (pose.heading == 'N') { pose.y += 1; }
        else if (pose.heading == 'S') { pose.y -= 1; }
    }

    // TurnLeft方法
    void ExecutorImpl::TurnLeft(void) noexcept
    {
        if (pose.heading == 'E') { pose.heading = 'N'; }
        else if (pose.heading == 'W') { pose.heading = 'S'; }
        else if (pose.heading == 'N') { pose.heading = 'W'; }
        else if (pose.heading == 'S') { pose.heading = 'E'; }
    }

    // TurnRight方法
    void ExecutorImpl::TurnRight(void) noexcept
    {
        if (pose.heading == 'E') { pose.heading = 'S'; }
        else if (pose.heading == 'W') { pose.heading = 'N'; }
        else if (pose.heading == 'N') { pose.heading = 'E'; }
        else if (pose.heading == 'S') { pose.heading = 'W'; }
    }

    // Fast方法
    void ExecutorImpl::Fast(void) noexcept
    {
        isfast = !isfast;
    }

    // isFast方法
    bool ExecutorImpl::isFast(void) const noexcept
    {
        return isfast;
    }
}