#pragma once
#include "Point.hpp"

namespace adas
{
    class Direction final
    {
    public:
        // 根据方向字符串获取方向
        static const Direction& GetDirection(const char heading) noexcept;
    
    public:
        Direction(const unsigned index, const char heading) noexcept;

    public:
        const Point& Move(void) const noexcept;
        const Direction& LeftOne(void) const noexcept;
        const Direction& RightOne(void) const noexcept;
        const char GetHeading(void) const noexcept; // 获取方向字符

    private:
        unsigned index; // 方向索引 0 1 2 3
        char heading;   // 方向字符 E S W N
    };
}