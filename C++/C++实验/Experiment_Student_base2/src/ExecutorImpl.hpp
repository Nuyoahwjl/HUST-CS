#pragma once

#include "Executor.hpp"
#include <string>

namespace adas
{
    /*
        Executor的具体实现
    */
    class ExecutorImpl : public Executor
    {
    public:
        // 构造函数
        explicit ExecutorImpl(const Pose &pose) noexcept;
        // 默认析构函数
        ~ExecutorImpl() noexcept = default;
        // 不能拷贝
        ExecutorImpl(const ExecutorImpl &) = delete;
        // 不能赋值
        ExecutorImpl &operator=(const ExecutorImpl &) = delete;

    public:
        // 查询当前汽车姿态,是父类抽象方法Query的具体实现
        Pose Query(void) const noexcept override;
        // 通过命令执行驾驶动作,是父类抽象方法Execute的具体实现
        void Execute(const std::string &command) noexcept override;

    private:
        // 当前汽车姿态
        Pose pose;
        // 是否为Fast状态
        bool isfast{false};
        // 移动方法
        void Move(void) noexcept;
        // 左转方法
        void TurnLeft(void) noexcept;
        // 右转方法
        void TurnRight(void) noexcept;
        // 改变Fast状态
        void Fast(void) noexcept;
        // 查询是否为Fast状态
        bool isFast(void) const noexcept;

        // 定义一个虚基类ICommand，完成DoOperate动作
        class ICommand
        {
        public:
            virtual void DoOperate(ExecutorImpl &executor) const noexcept = 0;
            virtual ~ICommand() noexcept = default;
        };
        // 定义一个嵌套类MoveCommand，完成Move动作
        class MoveCommand final : public ICommand
        {
        public:
            void DoOperate(ExecutorImpl &executor) const noexcept override
            {
                if(executor.isFast())
                    executor.Move();
                executor.Move();
            }
        };
        // 定义一个嵌套类TurnLeftCommand，完成TurnLeft动作
        class TurnLeftCommand final : public ICommand
        {
        public:
            void DoOperate(ExecutorImpl &executor) const noexcept override
            {
                if (executor.isFast())
                    executor.Move();
                executor.TurnLeft();
            }
        };
        // 定义一个嵌套类TurnRightCommand，完成TurnRight动作
        class TurnRightCommand final : public ICommand
        {
        public:
            void DoOperate(ExecutorImpl &executor) const noexcept override
            {
                if (executor.isFast())
                    executor.Move();
                executor.TurnRight();
            }
        };

        // // 定义一个嵌套类MoveCommand，完成Move动作
        // class MoveCommand final
        // {
        // public:
        //     void DoOperate(ExecutorImpl& executor) const noexcept
        //     {
        //         executor.Move();
        //     }
        // };
        // // 定义一个嵌套类TurnLeftCommand，完成TurnLeft动作
        // class TurnLeftCommand final
        // {
        // public:
        //     void DoOperate(ExecutorImpl& executor) const noexcept
        //     {
        //         executor.TurnLeft();
        //     }
        // };
        // // 定义一个嵌套类TurnRightCommand，完成TurnRight动作
        // class TurnRightCommand final
        // {
        // public:
        //     void DoOperate(ExecutorImpl& executor) const noexcept
        //     {
        //         executor.TurnRight();
        //     }
        // };

        // 定义一个嵌套类FastCommand，完成Fast动作
        class FastCommand final : public ICommand
        {
        public:
            void DoOperate(ExecutorImpl &executor) const noexcept override
            {
                executor.Fast();
            }
        };
    };
}