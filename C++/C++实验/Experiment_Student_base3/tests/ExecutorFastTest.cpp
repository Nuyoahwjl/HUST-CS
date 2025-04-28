#include <gtest/gtest.h>
#include <memory>
#include "Executor.hpp"
#include "PoseEq.hpp"

namespace adas
{
    // Fast状态下的测试用例1
    TEST(ExecutorFastTest, should_return_x_plus_2_given_status_is_fast_command_is_M_and_facing_is_E)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));

        // when
        executor->Execute("FM");

        // then
        const Pose target = {2, 0, 'E'};
        ASSERT_EQ(executor->Query(), target);
    }

    // Fast状态下的测试用例2
    TEST(ExecutorFastTest, should_return_N_and_x_plus_1_given_status_is_fast_command_is_L_and_facing_is_E)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));

        // when
        executor->Execute("FL");

        // then
        const Pose target = {1, 0, 'N'};
        ASSERT_EQ(executor->Query(), target);
    }

    // Fast状态下的测试用例3
    TEST(ExecutorFastTest, should_return_S_and_x_plus_1_given_status_is_fast_command_is_R_and_facing_is_E)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));

        // when
        executor->Execute("FR");

        // then
        const Pose target = {1, 0, 'S'};
        ASSERT_EQ(executor->Query(), target);
    }

    // Fast状态下的测试用例4
    TEST(ExecutorFastTest, should_return_y_plus_1_given_command_is_FFM_and_facing_is_N)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'N'}));

        // when
        executor->Execute("FFM");

        // then
        const Pose target = {0, 1, 'N'};
        ASSERT_EQ(executor->Query(), target);
    }
}