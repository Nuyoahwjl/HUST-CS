#include <gtest/gtest.h>
#include <memory>
// #include <tuple>
#include "Executor.hpp"
#include "PoseEq.hpp"

namespace adas
{
    // 重载Pose的==
    // bool operator==(const Pose &lhs, const Pose &rhs)
    // {
    //     return std::tie(lhs.x, lhs.y, lhs.heading) == std::tie(rhs.x, rhs.y, rhs.heading);
    // }

    // 测试用例1
    TEST(ExecutorTest, should_return_init_pose_when_without_command)
    {
        // given给定测试条件
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'})); // 给定初始姿态

        // when

        // then
        const Pose target = {0, 0, 'E'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例2
    TEST(ExecutorTest, should_return_default_pose_when_init_and_command)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor()); // 没有给定初始姿态

        // when

        // then
        const Pose target = {0, 0, 'N'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例3, 测试Execute方法,朝向E,起点(0,0),执行M命令,期望结果为(1,0,E)
    TEST(ExecutorTest, should_return_x_plus_1_given_command_is_M_and_facing_is_E)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));

        // when
        executor->Execute("M");

        // then
        const Pose target = {1, 0, 'E'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例4, 测试Execute方法,朝向W,起点(0,0),执行M命令,期望结果为(-1,0,W)
    TEST(ExecutorTest, should_return_x_minus_1_given_command_is_M_and_facing_is_W)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'W'}));

        // when
        executor->Execute("M");

        // then
        const Pose target = {-1, 0, 'W'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例5, 测试Execute方法,朝向N,起点(0,0),执行M命令,期望结果为(0,1,N)
    TEST(ExecutorTest, should_return_x_plus_1_given_command_is_M_and_facing_is_N)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'N'}));

        // when
        executor->Execute("M");

        // then
        const Pose target = {0, 1, 'N'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例6, 测试Execute方法,朝向S,起点(0,0),执行M命令,期望结果为(0,-1,S)
    TEST(ExecutorTest, should_return_x_minus_1_given_command_is_M_and_facing_is_S)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'S'}));

        // when
        executor->Execute("M");

        // then
        const Pose target = {0, -1, 'S'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例7, 测试Execute方法,朝向E,起点(0,0),执行L命令,期望结果为(0,0,N)
    TEST(ExecutorTest, should_return_facing_N_given_command_is_L_and_facing_is_E)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));

        // when
        executor->Execute("L");

        // then
        const Pose target = {0, 0, 'N'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例8, 测试Execute方法,朝向W,起点(0,0),执行L命令,期望结果为(0,0,S)
    TEST(ExecutorTest, should_return_facing_S_given_command_is_L_and_facing_is_W)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'W'}));

        // when
        executor->Execute("L");

        // then
        const Pose target = {0, 0, 'S'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例9, 测试Execute方法,朝向N,起点(0,0),执行L命令,期望结果为(0,0,W)
    TEST(ExecutorTest, should_return_facing_W_given_command_is_L_and_facing_is_N)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'N'}));

        // when
        executor->Execute("L");

        // then
        const Pose target = {0, 0, 'W'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例10, 测试Execute方法,朝向S,起点(0,0),执行L命令,期望结果为(0,0,E)
    TEST(ExecutorTest, should_return_facing_E_given_command_is_L_and_facing_is_S)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'S'}));

        // when
        executor->Execute("L");

        // then
        const Pose target = {0, 0, 'E'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例11, 测试Execute方法,朝向E,起点(0,0),执行R命令,期望结果为(0,0,S)
    TEST(ExecutorTest, should_return_facing_S_given_command_is_R_and_facing_is_E)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));

        // when
        executor->Execute("R");

        // then
        const Pose target = {0, 0, 'S'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例12, 测试Execute方法,朝向W,起点(0,0),执行R命令,期望结果为(0,0,N)
    TEST(ExecutorTest, should_return_facing_N_given_command_is_R_and_facing_is_W)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'W'}));

        // when
        executor->Execute("R");

        // then
        const Pose target = {0, 0, 'N'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例13, 测试Execute方法,朝向N,起点(0,0),执行R命令,期望结果为(0,0,E)
    TEST(ExecutorTest, should_return_facing_E_given_command_is_R_and_facing_is_N)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'N'}));

        // when
        executor->Execute("R");

        // then
        const Pose target = {0, 0, 'E'};
        ASSERT_EQ(executor->Query(), target);
    }

    // 测试用例14, 测试Execute方法,朝向S,起点(0,0),执行R命令,期望结果为(0,0,W)
    TEST(ExecutorTest, should_return_facing_W_given_command_is_R_and_facing_is_S)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'S'}));

        // when
        executor->Execute("R");

        // then
        const Pose target = {0, 0, 'W'};
        ASSERT_EQ(executor->Query(), target);
    }
}