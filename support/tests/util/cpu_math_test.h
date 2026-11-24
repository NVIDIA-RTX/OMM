/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <cfenv>
#include <gtest/gtest.h>
#if defined(_WIN32)
#include <float.h>
#endif

class CpuMathTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(std::fegetenv(&m_previousEnvironment), 0);
#if defined(_WIN32)
        ASSERT_EQ(_controlfp_s(&m_previousControl, 0, 0), 0);
#endif
        m_saved = true;
        // Mask exceptions for NaN/infinity probes and select nearest-even with
        // gradual underflow. glibc's default environment also clears FTZ/DAZ.
        ASSERT_EQ(std::fesetenv(FE_DFL_ENV), 0);
#if defined(_WIN32)
        // Windows exposes denormal controls separately from the C environment.
        unsigned int control;
        ASSERT_EQ(_controlfp_s(&control, _DN_SAVE, _MCW_DN), 0);
#endif
        ASSERT_EQ(std::fegetround(), FE_TONEAREST);
    }

    void TearDown() override
    {
        if (!m_saved)
            return;
        EXPECT_EQ(std::fesetenv(&m_previousEnvironment), 0);
#if defined(_WIN32)
        unsigned int control;
        EXPECT_EQ(_controlfp_s(&control, m_previousControl & _MCW_DN, _MCW_DN), 0);
#endif
    }

private:
    std::fenv_t m_previousEnvironment{};
    bool m_saved = false;
#if defined(_WIN32)
    unsigned int m_previousControl = 0;
#endif
};
