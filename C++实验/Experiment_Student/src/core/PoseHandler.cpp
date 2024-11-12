#include "PoseHandler.hpp"

namespace adas
{
    PoseHandler::PoseHandler(const Pose &pose) noexcept 
        // 初始化Pose改为初始化Point和Direction
        : point(pose.x,pose.y)
        ,facing(&Direction::GetDirection(pose.heading)) {}

    void PoseHandler::Forward(void) noexcept
    {
        // if (pose.heading == 'E') { pose.x += 1; }
        // else if (pose.heading == 'W') { pose.x -= 1; }
        // else if (pose.heading == 'N') { pose.y += 1; }
        // else if (pose.heading == 'S') { pose.y -= 1; }
        point += facing->Move();
    }

    void PoseHandler::Backward(void) noexcept
    {
        point -= facing->Move();
    }

    void PoseHandler::TurnLeft(void) noexcept
    {
        // if (pose.heading == 'E') { pose.heading = 'N'; }
        // else if (pose.heading == 'W') { pose.heading = 'S'; }
        // else if (pose.heading == 'N') { pose.heading = 'W'; }
        // else if (pose.heading == 'S') { pose.heading = 'E'; }
        facing = &(facing->LeftOne());
    }

    void PoseHandler::TurnRight(void) noexcept
    {
        // if (pose.heading == 'E') { pose.heading = 'S'; }
        // else if (pose.heading == 'W') { pose.heading = 'N'; }
        // else if (pose.heading == 'N') { pose.heading = 'E'; }
        // else if (pose.heading == 'S') { pose.heading = 'W'; }
        facing = &(facing->RightOne());
    }

    void PoseHandler::Fast(void) noexcept
    {
        isfast = !isfast;
    }

    void PoseHandler::Reverse(void) noexcept
    {
        isreverse = !isreverse;
    }

    bool PoseHandler::isFast(void) const noexcept
    {
        return isfast;
    }

    bool PoseHandler::isReverse(void) const noexcept
    {
        return isreverse;
    }

    Pose PoseHandler::Query(void) const noexcept
    {
        return {point.GetX(),point.GetY(),facing->GetHeading()};
    }
}