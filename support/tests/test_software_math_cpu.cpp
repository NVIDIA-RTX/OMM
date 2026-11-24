/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "util/cpu_math_test.h"
#include "util/omm_software_math.h"

namespace
{
    using Float = std::numeric_limits<float>;

    uint32_t ReferenceBits(float value)
    {
        // CPU NaN signs/payloads vary; the software utility canonicalizes them.
        return std::isnan(value) ? 0x7FC00000u : std::bit_cast<uint32_t>(value);
    }
}

TEST_F(CpuMathTest, SoftwareSqrt)
{
    const float inputs[] = {
        0.0f,                   // Positive zero.
        -0.0f,                  // Negative zero; its sign must be preserved.
        1.0f,                   // Exact unit result.
        2.0f,                   // Irrational result requiring rounding.
        4.0f,                   // Perfect square.
        0.25f,                  // Fractional perfect square.
        0.5f,                   // Fractional input requiring rounding.
        3.0f,                   // Another irrational result.
        9.0f,                   // Odd perfect square.
        1.0e-20f,               // Small finite input.
        1.0e20f,                // Large finite input.
        -1.0f,                  // Negative finite input produces NaN.
        Float::denorm_min(),    // Smallest positive subnormal.
        Float::min(),           // Smallest positive normal.
        Float::max(),           // Largest finite value.
        Float::infinity(),      // Positive infinity.
        -Float::infinity(),     // Negative infinity produces NaN.
        Float::quiet_NaN(),     // Quiet NaN.
        Float::signaling_NaN()  // Signaling NaN becomes quiet.
    };

    for (float input : inputs)
    {
        const float expected = std::sqrt(input);
        const float actual = omm::util::OmmSoftwareSqrt(input);
        EXPECT_EQ(std::bit_cast<uint32_t>(actual), ReferenceBits(expected)) << "sqrt(" << input << ')';
    }
}

TEST_F(CpuMathTest, SoftwareDivide)
{
    const float inputs[][2] = {
        { 6.0f, 2.0f },                  // Exact result.
        { 1.0f, 3.0f },                  // Rounded result.
        { -1.0f, 3.0f },                 // Negative numerator.
        { 1.0f, -3.0f },                 // Negative denominator.
        { -1.0f, -3.0f },                // Two negative operands.
        { 7.0f, 10.0f },                 // Rounded fractional quotient.
        { 0.5f, 0.25f },                 // Exact quotient of fractions.
        { -0.0f, -2.0f },                // Two negative operands produce positive zero.
        { 1.0f, -Float::infinity() },     // Finite divided by negative infinity.
        { Float::denorm_min(), Float::denorm_min() }, // Two subnormals produce one.
        { 0.0f, 2.0f },                  // Positive zero.
        { -0.0f, 2.0f },                 // Negative zero.
        { 1.0f, 0.0f },                  // Positive infinity from division by zero.
        { 1.0f, -0.0f },                 // Negative infinity from division by zero.
        { 0.0f, 0.0f },                  // Invalid zero divided by zero.
        { 1.0f, Float::infinity() },      // Finite divided by infinity.
        { Float::infinity(), 2.0f },      // Infinity divided by finite.
        { Float::infinity(), Float::infinity() }, // Invalid infinity divided by infinity.
        { Float::quiet_NaN(), 1.0f },     // NaN numerator.
        { 1.0f, Float::quiet_NaN() },     // NaN denominator.
        { Float::signaling_NaN(), 1.0f }, // Signaling NaN becomes quiet.
        { Float::denorm_min(), 1.0f },    // Preserve a subnormal operand/result.
        { Float::denorm_min(), 2.0f },    // Underflow rounds to even zero.
        { Float::min(), 2.0f },           // Normal input produces a subnormal.
        { Float::max(), 0.5f }           // Overflow produces infinity.
    };

    for (const auto& input : inputs)
    {
        const float expected = input[0] / input[1];
        const float actual = omm::util::OmmSoftwareDivide(input[0], input[1]);
        EXPECT_EQ(std::bit_cast<uint32_t>(actual), ReferenceBits(expected))
            << input[0] << " / " << input[1];
    }
}
