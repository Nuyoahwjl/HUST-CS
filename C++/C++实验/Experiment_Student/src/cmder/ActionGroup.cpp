#include "ActionGroup.hpp"
#include "CmderFactory.hpp"
#include <vector>
#include <algorithm>

namespace adas
{
    class ForwardAction final
    {
    public:
        void operator()(PoseHandler& poseHandler) const noexcept
        {
            poseHandler.Forward();
        }
    };

    class BackwardAction final
    {
    public:
        void operator()(PoseHandler& poseHandler) const noexcept
        {
            poseHandler.Backward();
        }
    };

    class TurnLeftAction final
    {
    public:
        void operator()(PoseHandler& poseHandler) const noexcept
        {
            poseHandler.TurnLeft();
        }
    };

    class ReverseTurnLeftAction final
    {
    public:
        void operator()(PoseHandler& poseHandler) const noexcept
        {
            poseHandler.TurnRight();
        }
    };

    class TurnRightAction final
    {
    public:
        void operator()(PoseHandler& poseHandler) const noexcept
        {
            poseHandler.TurnRight();
        }
    };

    class ReverseTurnRightAction final
    {
    public:
        void operator()(PoseHandler& poseHandler) const noexcept
        {
            poseHandler.TurnLeft();
        }
    };

    class BeFastAction final
    {
    public:
        void operator()(PoseHandler& poseHandler) const noexcept
        {
            poseHandler.Fast();
        }
    };

    class BeReverseAction final
    {
    public:
        void operator()(PoseHandler& poseHandler) const noexcept
        {
            poseHandler.Reverse();
        }
    };

    ActionGroup::ActionGroup(const std::list<ActionType>& actions) noexcept : actions(actions) {}

    void ActionGroup::PushAction(const ActionType& ActionType) noexcept
    {
        actions.push_back(ActionType);
    }

    void ActionGroup::DoOperate(PoseHandler& poseHandler) const noexcept
    {
        static std::vector<std::function<void(PoseHandler& poseHandler)>> actionVec = {
            ForwardAction(),
            BackwardAction(),
            TurnLeftAction(),
            ReverseTurnLeftAction(),
            TurnRightAction(),
            ReverseTurnRightAction(),
            BeFastAction(),
            BeReverseAction(),
        };

        std::for_each(
            actions.begin(),
            actions.end(),
            [&poseHandler](const ActionType& actionType) mutable noexcept
            {
                actionVec[static_cast<uint16_t>(actionType)](poseHandler);
            });
    }

    ActionGroup& ActionGroup::operator+=(const ActionGroup& rhs) noexcept
    {
        actions.insert(actions.end(), rhs.actions.begin(), rhs.actions.end());
        return *this;
    }
    
}