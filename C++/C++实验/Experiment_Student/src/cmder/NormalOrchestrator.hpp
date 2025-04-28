#pragma once

#include "CmderOrchestrator.hpp"

namespace adas
{
    class NormalOrchestrator : public CmderOrchestrator
    {
    public:
        ActionGroup Move(const PoseHandler &poseHandler) const noexcept override
        {
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler); // 加速状态下，先移动一步
            actionGroup.PushAction(GetStepAction(poseHandler)); // 不管是否是加速状态，移动一步
            return actionGroup;
        }

        ActionGroup TurnLeft(const PoseHandler &poseHandler) const noexcept override
        {
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler);
            actionGroup.PushAction(poseHandler.isReverse() ? ActionType::REVERSE_TURNLEFT_ACTION : ActionType::TURNLEFT_ACTION); 
            return actionGroup;
        }

        ActionGroup TurnRight(const PoseHandler &poseHandler) const noexcept override
        {
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler);
            actionGroup.PushAction(poseHandler.isReverse() ? ActionType::REVERSE_TURNRIGHT_ACTION : ActionType::TURNRIGHT_ACTION);
            return actionGroup;
        }

        ActionGroup TurnRound(const PoseHandler &poseHandler) const noexcept override
        {
            if(poseHandler.isReverse())
            {
                return ActionGroup();
            }
            else
            {
                ActionGroup actionGroup;
                actionGroup += OnFast(poseHandler);
                actionGroup.PushAction(ActionType::TURNLEFT_ACTION);
                actionGroup.PushAction(ActionType::FORWARD_1_STEP_ACTION);
                actionGroup.PushAction(ActionType::TURNLEFT_ACTION);
                return actionGroup;
            }
        }

    protected:
        ActionType GetStepAction(const PoseHandler &poseHandler) const noexcept
        {
            return poseHandler.isReverse() ? ActionType::BACKWARD_1_STEP_ACTION : ActionType::FORWARD_1_STEP_ACTION;
        }
        ActionGroup OnFast(const PoseHandler &poseHandler) const noexcept
        {
            if(poseHandler.isFast())
            {
                return ActionGroup({GetStepAction(poseHandler)});
            }
            return ActionGroup();
        }
    };

}