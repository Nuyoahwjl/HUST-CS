#pragma once
namespace adas
{
    class Point final
    {
    public:
        Point(const int x, const int y) noexcept;
        Point(const Point &rhs) noexcept; // 拷贝构造
        Point &operator=(const Point &rhs) noexcept; // 拷贝赋值
        Point &operator+=(const Point &rhs) noexcept; // 移动
        Point &operator-=(const Point &rhs) noexcept; // 移动

    public:
        int GetX(void) const noexcept; // 获取x坐标
        int GetY(void) const noexcept; // 获取y坐标

    private:
        int x; // x坐标
        int y; // y坐标
    };
}