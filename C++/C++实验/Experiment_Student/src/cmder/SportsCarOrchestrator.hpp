#pragma once

#include "NormalOrchestrator.hpp"

namespace adas
{
    class SportsCarOrchestrator : public NormalOrchestrator
    {
    public:
        ActionGroup Move(const PoseHandler &poseHandler) const noexcept override
        {
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler);
            actionGroup += OnFast(poseHandler); // 加速状态下，先移动两步
            actionGroup.PushAction(GetStepAction(poseHandler));
            actionGroup.PushAction(GetStepAction(poseHandler)); // 不管是否是加速状态，移动两步
            return actionGroup;
        }

        ActionGroup TurnLeft(const PoseHandler &poseHandler) const noexcept override
        {
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler); // 加速状态下，先移动一步
            actionGroup.PushAction(poseHandler.isReverse() ? ActionType::REVERSE_TURNLEFT_ACTION : ActionType::TURNLEFT_ACTION);
            actionGroup.PushAction(GetStepAction(poseHandler));
            return actionGroup;
        }

        ActionGroup TurnRight(const PoseHandler &poseHandler) const noexcept override
        {
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler); // 加速状态下，先移动一步
            actionGroup.PushAction(poseHandler.isReverse() ? ActionType::REVERSE_TURNRIGHT_ACTION : ActionType::TURNRIGHT_ACTION);
            actionGroup.PushAction(GetStepAction(poseHandler));
            return actionGroup;
        }
    };
}