#pragma once

#include "ActionGroup.hpp"

namespace adas
{
    class CmderOrchestrator 
    {
    public:
        virtual ~CmderOrchestrator(void) = default;

    public:
        virtual ActionGroup Move(const PoseHandler& poseHandler) const noexcept = 0;
        virtual ActionGroup TurnLeft(const PoseHandler& poseHandler) const noexcept = 0;
        virtual ActionGroup TurnRight(const PoseHandler& poseHandler) const noexcept = 0;
        virtual ActionGroup TurnRound(const PoseHandler& poseHandler) const noexcept = 0;
    };
}