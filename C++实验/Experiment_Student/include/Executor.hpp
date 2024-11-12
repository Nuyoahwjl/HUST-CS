#pragma once
#include <string>
namespace adas
{
    /* 车辆枚举类型 */
    enum class ExecutorType
    {
        NORMAL, 
        SPORTS_CAR,
        BUS,
    };
    
    /* 汽车姿态 */
    struct Pose
    {
        int x;        // x坐标
        int y;        // y坐标
        char heading; // 'N', 'E', 'S', 'W'代表四个方向
    };

    /*
        驾驶动作执行器接口
    */
    class Executor
    {
    public:
        // Caller should delete *executor when it is no longer needed.
        // static Executor *NewExecutor(const Pose &pose = {0, 0, 'N'}) noexcept;
        static Executor *NewExecutor(const Pose &pose = {0, 0, 'N'},
            const ExecutorType executorType = ExecutorType::NORMAL) noexcept;

    public:
        // 默认构造函数
        Executor(void) = default;
        // 默认析构函数
        virtual ~Executor(void) = default;

        // 不允许拷贝
        Executor(const Executor &) = delete;
        // 不允许赋值
        Executor &operator=(const Executor &) = delete;

    public:
        // 查询当前汽车姿态，纯虚函数，留给子类具体实现
        virtual Pose Query(void) const noexcept = 0;
        // 通过命令执行驾驶动作，纯虚函数，留给子类具体实现
        virtual void Execute(const std::string &command) noexcept = 0;
    }; // 含有纯虚函数的类称为抽象类，不能实例化，只能作为基类
}