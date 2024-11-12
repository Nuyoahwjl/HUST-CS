#include "ExecutorImpl.hpp"
// #include <new>
// #include <memory>
// #include "Command.hpp"
// #include <unordered_map>
#include "../src/cmder/CmderFactory.hpp"
#include "../src/core/Singleton.hpp"
#include <algorithm>
#include "../src/cmder/NormalOrchestrator.hpp"
#include "../src/cmder/SportsCarOrchestrator.hpp"
#include "../src/cmder/BusOrchestrator.hpp"

namespace adas
{
    // 初始化ExecutorImpl的pose成员变量
    ExecutorImpl::ExecutorImpl(const Pose &pose, CmderOrchestrator *orchestrator) noexcept : poseHandler(pose), orchestrator(orchestrator) {}

    // Query方法
    Pose ExecutorImpl::Query(void) const noexcept
    {
        return poseHandler.Query();
    }

    // NewExecutor方法
    Executor *Executor::NewExecutor(const Pose &pose, const ExecutorType executorType) noexcept
    {
        CmderOrchestrator *orchestrator = nullptr;
        switch(executorType)
        {
            case ExecutorType::NORMAL:
                orchestrator = new (std::nothrow) NormalOrchestrator();
                break;
            case ExecutorType::SPORTS_CAR:
                orchestrator = new (std::nothrow) SportsCarOrchestrator();
                break;
            case ExecutorType::BUS:
                orchestrator = new (std::nothrow) BusOrchestrator();
                break;
        }
        return new (std::nothrow) ExecutorImpl(pose,orchestrator); // c++17
    }

    // Execute方法
    void ExecutorImpl::Execute(const std::string &command) noexcept
    {
        // // 表驱动
        // // std::unordered_map<char, std::unique_ptr<ICommand>> cmderMap;
        // // std::unordered_map<char, std::function<void(PoseHandler & PoseHandler)>> cmderMap;
        // std::unordered_map<char, std::function<void(PoseHandler & PoseHandler)>> cmderMap {
        //     {'M', MoveCommand()},
        //     {'L', TurnLeftCommand()},
        //     {'R', TurnRightCommand()},
        //     {'F', FastCommand()},
        //     {'B', ReverseCommand()}
        // };
        //
        // // 建立操作与命令的映射
        // // cmderMap.emplace('M', std::make_unique<MoveCommand>());
        // // cmderMap.emplace('L', std::make_unique<TurnLeftCommand>());
        // // cmderMap.emplace('R', std::make_unique<TurnRightCommand>());
        // // cmderMap.emplace('F', std::make_unique<FastCommand>());
        //
        // // MoveCommand moveCommand;
        // // cmderMap.emplace('M', moveCommand.operate);
        // // TurnLeftCommand turnLeftCommand;
        // // cmderMap.emplace('L', turnLeftCommand.operate);
        // // TurnRightCommand turnRightCommand;
        // // cmderMap.emplace('R', turnRightCommand.operate);
        // // FastCommand fastCommand;
        // // cmderMap.emplace('F', fastCommand.operate);
        //
        // // cmderMap.emplace('M', MoveCommand());
        // // cmderMap.emplace('L', TurnLeftCommand());
        // // cmderMap.emplace('R', TurnRightCommand());
        // // cmderMap.emplace('F', FastCommand());
        //
        // // 执行命令
        // for (const auto cmd : command)
        // {
        //     // 根据操作查找表驱动
        //     const auto it = cmderMap.find(cmd);
        //     // 如果找到表驱动，执行对应操作
        //     if (it != cmderMap.end())
        //         it->second(poseHandler);
        //         // it->second->DoOperate(poseHandler);
        //         // cmderMap[cmd]->DoOperate(poseHandler);
        // }
        const auto cmders = Singleton<CmderFactory>::Instance().GetCmders(command);
        // std::for_each(
        //     cmders.begin(),
        //     cmders.end(),
        //     [this](const std::function<void(PoseHandler & poseHandler)> &cmder) noexcept
        //     {
        //         cmder(poseHandler);
        //     });
        std::for_each(
            cmders.begin(),
            cmders.end(),
            [this](const Cmder &cmder) noexcept
            {
                // cmder(poseHandler);
                cmder(poseHandler, *orchestrator).DoOperate(poseHandler);
            }
        );
    }

    // // Execute方法
    // void ExecutorImpl::Execute(const std::string &command) noexcept
    // {
    //     // for (const auto cmd : command)
    //     // {
    //     //     if (cmd == 'M')
    //     //     {
    //     //         // Move();
    //     //         std::unique_ptr<MoveCommand> cmder=std::make_unique<MoveCommand>();
    //     //         cmder->DoOperate(*this);
    //     //     }
    //     //     else if (cmd == 'L')
    //     //     {
    //     //         // TurnLeft();
    //     //         std::unique_ptr<TurnLeftCommand> cmder=std::make_unique<TurnLeftCommand>();
    //     //         cmder->DoOperate(*this);
    //     //     }
    //     //     else if (cmd == 'R')
    //     //     {
    //     //         // TurnRight();
    //     //         std::unique_ptr<TurnRightCommand> cmder=std::make_unique<TurnRightCommand>();
    //     //         cmder->DoOperate(*this);
    //     //     }
    //     // }
    //     for (const auto cmd : command)
    //     {
    //         std::unique_ptr<ICommand> cmder;
    //         if (cmd == 'M')
    //             cmder = std::make_unique<MoveCommand>();
    //         else if (cmd == 'L')
    //             cmder = std::make_unique<TurnLeftCommand>();
    //         else if (cmd == 'R')
    //             cmder = std::make_unique<TurnRightCommand>();
    //         else if (cmd == 'F')
    //             cmder = std::make_unique<FastCommand>();
    //         if(cmder)
    //             cmder->DoOperate(poseHandler);
    //     }
    // }

    // // Move方法
    // void ExecutorImpl::Move(void) noexcept
    // {
    //     if (pose.heading == 'E') { pose.x += 1; }
    //     else if (pose.heading == 'W') { pose.x -= 1; }
    //     else if (pose.heading == 'N') { pose.y += 1; }
    //     else if (pose.heading == 'S') { pose.y -= 1; }
    // }
    // // TurnLeft方法
    // void ExecutorImpl::TurnLeft(void) noexcept
    // {
    //     if (pose.heading == 'E') { pose.heading = 'N'; }
    //     else if (pose.heading == 'W') { pose.heading = 'S'; }
    //     else if (pose.heading == 'N') { pose.heading = 'W'; }
    //     else if (pose.heading == 'S') { pose.heading = 'E'; }
    // }
    // // TurnRight方法
    // void ExecutorImpl::TurnRight(void) noexcept
    // {
    //     if (pose.heading == 'E') { pose.heading = 'S'; }
    //     else if (pose.heading == 'W') { pose.heading = 'N'; }
    //     else if (pose.heading == 'N') { pose.heading = 'E'; }
    //     else if (pose.heading == 'S') { pose.heading = 'W'; }
    // }
    // // Fast方法
    // void ExecutorImpl::Fast(void) noexcept
    // {
    //     isfast = !isfast;
    // }
    // // isFast方法
    // bool ExecutorImpl::isFast(void) const noexcept
    // {
    //     return isfast;
    // }

}