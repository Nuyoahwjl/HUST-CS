#pragma once
#include "../src/core/PoseHandler.hpp"
#include <functional>
// #include "ActionGroup.hpp"
#include "CmderOrchestrator.hpp"

namespace adas
{
    // // 定义一个虚基类ICommand，完成DoOperate动作
    // class ICommand
    // {
    // public:
    //     virtual void DoOperate(PoseHandler &poseHandler) const noexcept = 0;
    //     virtual ~ICommand() noexcept = default;
    // };
    // // 定义一个嵌套类MoveCommand，完成Move动作
    // class MoveCommand final : public ICommand
    // {
    // public:
    //     void DoOperate(PoseHandler &poseHandler) const noexcept override
    //     {
    //         if (poseHandler.isFast())
    //             poseHandler.Move();
    //         poseHandler.Move();
    //     }
    // };
    // // 定义一个嵌套类TurnLeftCommand，完成TurnLeft动作
    // class TurnLeftCommand final : public ICommand
    // {
    // public:
    //     void DoOperate(PoseHandler &poseHandler) const noexcept override
    //     {
    //         if (poseHandler.isFast())
    //             poseHandler.Move();
    //         poseHandler.TurnLeft();
    //     }
    // };
    // // 定义一个嵌套类TurnRightCommand，完成TurnRight动作
    // class TurnRightCommand final : public ICommand
    // {
    // public:
    //     void DoOperate(PoseHandler &poseHandler) const noexcept override
    //     {
    //         if (poseHandler.isFast())
    //             poseHandler.Move();
    //         poseHandler.TurnRight();
    //     }
    // };
    // // 定义一个嵌套类FastCommand，完成Fast动作
    // class FastCommand final : public ICommand
    // {
    // public:
    //     void DoOperate(PoseHandler &poseHandler) const noexcept override
    //     {
    //         poseHandler.Fast();
    //     }
    // };

    class MoveCommand final
    {
    public:
        // // 定义函数对象operate，接受参数PoseHandler，返回void
        // const std::function<void(PoseHandler &PoseHandler)> operate = [](PoseHandler &poseHandler) noexcept
        // {
        //     if (poseHandler.isFast())
        //         poseHandler.Move();
        //     poseHandler.Move();
        // };

        // void operator()(PoseHandler &poseHandler) const noexcept
        // {
        //     if(poseHandler.isFast())
        //     {
        //         if(poseHandler.isReverse())
        //             poseHandler.Backward();
        //         else
        //             poseHandler.Forward();
        //     }
        //     if(poseHandler.isReverse())
        //         poseHandler.Backward();
        //     else
        //         poseHandler.Forward();
        // }

        // ActionGroup operator()(PoseHandler &poseHandler) const noexcept
        // {
        //     ActionGroup actionGroup;
        //     const auto action = poseHandler.isReverse() ? ActionType::BACKWARD_1_STEP_ACTION : ActionType::FORWARD_1_STEP_ACTION;
        //     if (poseHandler.isFast())
        //     {
        //         actionGroup.PushAction(action);
        //     }
        //     actionGroup.PushAction(action);
        //     return actionGroup;
        // }

        ActionGroup operator()(const PoseHandler &poseHandler,const CmderOrchestrator &orchestrator) const noexcept
        {
            return orchestrator.Move(poseHandler);
        }
    };

    class TurnLeftCommand final
    {
    public:
        // // 定义函数对象operate，接受参数PoseHandler，返回void
        // const std::function<void(PoseHandler &PoseHandler)> operate = [](PoseHandler &poseHandler) noexcept
        // {
        //     if (poseHandler.isFast())
        //         poseHandler.Move();
        //     poseHandler.TurnLeft();
        // };

        // void operator()(PoseHandler &poseHandler) const noexcept
        // {
        //     if (poseHandler.isFast())
        //     {
        //         if(poseHandler.isReverse())
        //             poseHandler.Backward();
        //         else
        //             poseHandler.Forward();
        //     }
        //     if(poseHandler.isReverse())
        //         poseHandler.TurnRight();
        //     else
        //         poseHandler.TurnLeft();
        // }

        // ActionGroup operator()(PoseHandler &poseHandler) const noexcept
        // {
        //     ActionGroup actionGroup;
        //     const auto action = poseHandler.isReverse() ? ActionType::REVERSE_TURNLEFT_ACTION : ActionType::TURNLEFT_ACTION;
        //     if (poseHandler.isFast())
        //     {
        //         const auto action_ = poseHandler.isReverse() ? ActionType::BACKWARD_1_STEP_ACTION : ActionType::FORWARD_1_STEP_ACTION;
        //         actionGroup.PushAction(action_);
        //     }
        //     actionGroup.PushAction(action);
        //     return actionGroup;
        // }

        ActionGroup operator()(const PoseHandler &poseHandler,const CmderOrchestrator &orchestrator) const noexcept
        {
            return orchestrator.TurnLeft(poseHandler);
        }
    };

    class TurnRightCommand final
    {
    public:
        // // 定义函数对象operate，接受参数PoseHandler，返回void
        // const std::function<void(PoseHandler &PoseHandler)> operate = [](PoseHandler &poseHandler) noexcept
        // {
        //     if (poseHandler.isFast())
        //         poseHandler.Move();
        //     poseHandler.TurnRight();
        // };

        // void operator()(PoseHandler &poseHandler) const noexcept
        // {
        //     if (poseHandler.isFast())
        //     {
        //         if(poseHandler.isReverse())
        //             poseHandler.Backward();
        //         else
        //             poseHandler.Forward();
        //     }
        //     if(poseHandler.isReverse())
        //         poseHandler.TurnLeft();
        //     else
        //         poseHandler.TurnRight();
        // }

        // ActionGroup operator()(PoseHandler &poseHandler) const noexcept
        // {
        //     ActionGroup actionGroup;
        //     const auto action = poseHandler.isReverse() ? ActionType::REVERSE_TURNRIGHT_ACTION : ActionType::TURNRIGHT_ACTION;
        //     if (poseHandler.isFast())
        //     {
        //         const auto action_ = poseHandler.isReverse() ? ActionType::BACKWARD_1_STEP_ACTION : ActionType::FORWARD_1_STEP_ACTION;
        //         actionGroup.PushAction(action_);
        //     }
        //     actionGroup.PushAction(action);
        //     return actionGroup;
        // }

        ActionGroup operator()(const PoseHandler &poseHandler,const CmderOrchestrator &orchestrator) const noexcept
        {
            return orchestrator.TurnRight(poseHandler);
        }
    };

    class FastCommand final
    {
    public:
        // // 定义函数对象operate，接受参数PoseHandler，返回void
        // const std::function<void(PoseHandler &PoseHandler)> operate = [](PoseHandler &poseHandler) noexcept
        // {
        //     poseHandler.Fast();
        // };

        // void operator()(PoseHandler &poseHandler) const noexcept
        // {
        //     poseHandler.Fast();
        // }

        // ActionGroup operator()(PoseHandler &poseHandler) const noexcept
        // {
        //     ActionGroup actionGroup;
        //     actionGroup.PushAction(ActionType::BE_FAST_ACTION);
        //     return actionGroup;
        // }

        ActionGroup operator()(PoseHandler &poseHandler, const CmderOrchestrator &orchestrator) const noexcept
        {
            ActionGroup actionGroup;
            actionGroup.PushAction(ActionType::BE_FAST_ACTION);
            return actionGroup;
        }
    };

    class ReverseCommand final
    {
    public:
        // void operator()(PoseHandler &poseHandler) const noexcept
        // {
        //     poseHandler.Reverse();
        // }

        // ActionGroup operator()(PoseHandler &poseHandler) const noexcept
        // {
        //     ActionGroup actionGroup;
        //     actionGroup.PushAction(ActionType::BE_REVERSE_ACTION);
        //     return actionGroup;
        // }

        ActionGroup operator()(PoseHandler &poseHandler, const CmderOrchestrator &orchestrator) const noexcept
        {
            ActionGroup actionGroup;
            actionGroup.PushAction(ActionType::BE_REVERSE_ACTION);
            return actionGroup;
        }
    };

    class TurnRoundCommand final
    {
    public:
        // ActionGroup operator()(PoseHandler &poseHandler) const noexcept
        // {
        //     if (poseHandler.isReverse())
        //     {
        //         return ActionGroup(); // 倒车状态下，什么都不做
        //     }
        //     else
        //     {
        //         if (poseHandler.isFast()) // 快速状态下，四个原子Action
        //         {
        //             return ActionGroup({ActionType::FORWARD_1_STEP_ACTION,
        //                                 ActionType::TURNLEFT_ACTION,
        //                                 ActionType::FORWARD_1_STEP_ACTION,
        //                                 ActionType::TURNLEFT_ACTION});
        //         }
        //         else // 普通状态下，三个原子Action
        //         {
        //             return ActionGroup({ActionType::TURNLEFT_ACTION,
        //                                 ActionType::FORWARD_1_STEP_ACTION,
        //                                 ActionType::TURNLEFT_ACTION});
        //         }
        //     }
        // }

        ActionGroup operator()(PoseHandler &poseHandler,const CmderOrchestrator &orchestrator) const noexcept
        {
            return orchestrator.TurnRound(poseHandler);
        }
    };

}