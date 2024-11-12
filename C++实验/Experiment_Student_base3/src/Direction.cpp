#include "Direction.hpp"

namespace adas
{
    // 4种方向
    static const Direction directions[4] = {
        {0, 'E'},
        {1, 'S'},
        {2, 'W'},
        {3, 'N'}};

    // 4种前进坐标
    static const Point moves[4] = {
        {1, 0},  // E
        {0, -1}, // S
        {-1, 0}, // W
        {0, 1}   // N
    };

    const Direction &Direction::GetDirection(const char heading) noexcept
    {
        for (const auto &dir : directions)
        {
            if (dir.heading == heading)
            {
                return dir;
            }
        }
        return directions[3]; // 默认返回N
    }

    Direction::Direction(const unsigned index, const char heading) noexcept : index(index), heading(heading) {}

    const Point &Direction::Move(void) const noexcept
    {
        return moves[index];
    }

    const Direction &Direction::LeftOne(void) const noexcept
    {
        return directions[(index + 3) % 4];
    }

    const Direction &Direction::RightOne(void) const noexcept
    {
        return directions[(index + 1) % 4];
    }

    const char Direction::GetHeading(void) const noexcept
    {
        return heading;
    }
}