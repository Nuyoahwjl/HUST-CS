#pragma once

#include "../include/Executor.hpp"
#include "../src/core/PoseHandler.hpp"
#include <string>
#include <memory>
#include "../src/cmder/CmderOrchestrator.hpp"

namespace adas
{
    /*
        Executor的具体实现
    */
    class ExecutorImpl : public Executor
    {
    public:
        // 构造函数
        explicit ExecutorImpl(const Pose &pose, CmderOrchestrator *orchestrator) noexcept;
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
        
    // private:
    //     // 当前汽车姿态
    //     Pose pose;
    //     // 是否为Fast状态
    //     bool isfast{false};
    // public:
    //     // 移动方法
    //     void Move(void) noexcept;
    //     // 左转方法
    //     void TurnLeft(void) noexcept;
    //     // 右转方法
    //     void TurnRight(void) noexcept;
    //     // 改变Fast状态
    //     void Fast(void) noexcept;
    //     // 查询是否为Fast状态
    //     bool isFast(void) const noexcept;

    private:
        PoseHandler poseHandler; // 状态管理类
        std::unique_ptr<CmderOrchestrator> orchestrator;
    };
}