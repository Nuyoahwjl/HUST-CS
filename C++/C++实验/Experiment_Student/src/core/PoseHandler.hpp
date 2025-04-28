#pragma once
#include "../include/Executor.hpp"
#include "Direction.hpp"

namespace adas
{
    class PoseHandler final
    {
        public:
            PoseHandler(const Pose &pose) noexcept;
            PoseHandler(const PoseHandler &) = delete; // 不允许拷贝
            PoseHandler &operator=(const PoseHandler &) = delete; // 不允许赋值

        public:
            void Forward(void) noexcept; // 向前移动
            void Backward(void) noexcept; // 向后移动

            void TurnLeft(void) noexcept; // 左转
            void TurnRight(void) noexcept; // 右转

            void Fast(void) noexcept; // 切换加速模式
            void Reverse(void) noexcept; // 切换倒车模式

            bool isFast(void) const noexcept; // 查询是否为快速模式
            bool isReverse(void) const noexcept; // 查询是否为倒车模式

            Pose Query(void) const noexcept; // 查询当前汽车姿态

        private:
            // Pose pose; // 当前汽车姿态
            Point point; // 当前坐标
            const Direction* facing; // 当前方向
            bool isfast{false}; // 是否为快速模式
            bool isreverse{false}; // 是否为倒车模式
    };
}