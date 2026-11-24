/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <array>
#include <bit>
#include <cstdint>
#include <limits>

#include "util/cpu_math_test.h"
#include "util/hlsl_cpu_compat.h"

namespace software_sampling
{
    using namespace hlsl_cpu;
    // Strict compiler options preserve the operation order of HLSL's precise.
#define precise
#include "../../libraries/omm-lib/shaders/omm_software_sampling.hlsli"
#undef precise
}

namespace
{
    using namespace hlsl_cpu;
    using namespace software_sampling;
    using ResultBits = std::array<uint32_t, 4>;
    enum : uint { Wrap, Mirror, Clamp, Border };
    enum : uint { Nearest, Linear };

    // A 2x2 texture with a distinct value in every texel and channel.
    const float4 kTexels[] = {
        float4(1, 2, 3, 4),     float4(5, 6, 7, 8),
        float4(9, 10, 11, 12),  float4(13, 14, 15, 16)
    };
    const Texture2D<float4> kTexture{ kTexels, 2, 2 };
    const uint2 kSize(2, 2);
    constexpr float kBorder = 20.0f;

    ResultBits Bits(float4 value)
    {
        return { std::bit_cast<uint32_t>(value[0]), std::bit_cast<uint32_t>(value[1]),
            std::bit_cast<uint32_t>(value[2]), std::bit_cast<uint32_t>(value[3]) };
    }

    float4 FromBits(ResultBits bits)
    {
        return { std::bit_cast<float>(bits[0]), std::bit_cast<float>(bits[1]),
            std::bit_cast<float>(bits[2]), std::bit_cast<float>(bits[3]) };
    }
}

TEST_F(CpuMathTest, SoftwareSamplingNearest)
{
    struct TestCase
    {
        float2 uv;
        uint addressMode;
        float4 expected;
    };

    const TestCase cases[] = {
        { float2(0.25f, 0.25f), Wrap, kTexels[0] },
        { float2(0.75f, 0.25f), Wrap, kTexels[1] },
        { float2(0.25f, 0.75f), Wrap, kTexels[2] },
        { float2(0.75f, 0.75f), Wrap, kTexels[3] },
        { float2(-0.25f, 1.25f), Wrap, kTexels[1] },
        { float2(-0.25f, 1.25f), Mirror, kTexels[2] },
        { float2(-0.25f, 1.25f), Clamp, kTexels[2] },
        { float2(-0.25f, 1.25f), Border, float4(20, 20, 20, 20) },
        { float2(1.0f, 1.0f), Wrap, kTexels[0] },
        { float2(1.0f, 1.0f), Clamp, kTexels[3] }
    };

    for (const auto& test : cases)
        EXPECT_EQ(Bits(OmmSoftwareSampleLevel(kTexture, test.uv, kSize, test.addressMode, Nearest, kBorder)),
            Bits(test.expected)) << "UV " << test.uv.x << ',' << test.uv.y << ", mode " << test.addressMode;
}

TEST_F(CpuMathTest, SoftwareSamplingBilinear)
{
    struct TestCase
    {
        float2 uv;
        uint addressMode;
        float4 expected;
    };

    const TestCase cases[] = {
        // Texel center, center average, and unequal horizontal/vertical weights.
        { float2(0.25f, 0.25f), Clamp, float4(1, 2, 3, 4) },
        { float2(0.5f, 0.5f), Clamp, float4(7, 8, 9, 10) },
        { float2(0.375f, 0.625f), Clamp, float4(8, 9, 10, 11) },
        // Texture corners exercise addressing and mixed border/texel footprints.
        { float2(0, 0), Wrap, float4(7, 8, 9, 10) },
        { float2(0, 0), Mirror, float4(1, 2, 3, 4) },
        { float2(0, 0), Clamp, float4(1, 2, 3, 4) },
        { float2(0, 0), Border, float4(15.25f, 15.5f, 15.75f, 16) },
        { float2(1, 1), Border, float4(18.25f, 18.5f, 18.75f, 19) }
    };

    for (const auto& test : cases)
        EXPECT_EQ(Bits(OmmSoftwareSampleLevel(kTexture, test.uv, kSize, test.addressMode, Linear, kBorder)),
            Bits(test.expected)) << "UV " << test.uv.x << ',' << test.uv.y << ", mode " << test.addressMode;
}

TEST_F(CpuMathTest, SoftwareSamplingGather)
{
    struct TestCase
    {
        float2 uv;
        uint addressMode;
        uint channel;
        float4 expected;
    };

    // Gather order is bottom-left, bottom-right, top-right, top-left.
    const TestCase cases[] = {
        { float2(0.5f, 0.5f), Clamp, 0, float4(9, 13, 5, 1) },
        { float2(0.5f, 0.5f), Clamp, 1, float4(10, 14, 6, 2) },
        { float2(0.5f, 0.5f), Clamp, 2, float4(11, 15, 7, 3) },
        { float2(0.5f, 0.5f), Clamp, 3, float4(12, 16, 8, 4) },
        { float2(0, 0), Wrap, 0, float4(5, 1, 9, 13) },
        { float2(0, 0), Mirror, 0, float4(1, 1, 1, 1) },
        { float2(0, 0), Clamp, 0, float4(1, 1, 1, 1) },
        { float2(0, 0), Border, 0, float4(20, 1, 20, 20) }
    };

    for (const auto& test : cases)
        EXPECT_EQ(Bits(OmmSoftwareGather(kTexture, test.uv, kSize, test.addressMode, test.channel, kBorder)),
            Bits(test.expected)) << "UV " << test.uv.x << ',' << test.uv.y
                << ", mode " << test.addressMode << ", channel " << test.channel;
}

TEST_F(CpuMathTest, SoftwareSamplingAddressesExtremeCoordinates)
{
    struct TestCase
    {
        int coordinate;
        int expectedWrap;
        int expectedMirror;
        int expectedClamp;
    };

    // Expected indices in a three-texel row, including both int32 limits.
    const TestCase cases[] = {
        { -4, 2, 2, 0 },
        { -1, 2, 0, 0 },
        { 0, 0, 0, 0 },
        { 2, 2, 2, 2 },
        { 3, 0, 2, 2 },
        { 6, 0, 0, 2 },
        { std::numeric_limits<int>::min(), 1, 1, 0 },
        { std::numeric_limits<int>::max(), 1, 1, 2 }
    };

    for (const auto& test : cases)
    {
        SCOPED_TRACE(test.coordinate);
        EXPECT_EQ(OmmSoftwareAddressTexel(test.coordinate, 3, Wrap), test.expectedWrap);
        EXPECT_EQ(OmmSoftwareAddressTexel(test.coordinate, 3, Mirror), test.expectedMirror);
        EXPECT_EQ(OmmSoftwareAddressTexel(test.coordinate, 3, Clamp), test.expectedClamp);
    }
}

TEST_F(CpuMathTest, SoftwareSamplingPreservesConstantFiniteColors)
{
    const float4 colors[] = {
        FromBits({ 0x3F450F29u, 0x3EAAAAABu, 0x3F123457u, 0x3E4CCCCDu }),
        float4(std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(),
            std::numeric_limits<float>::max(), 0.0f),
        float4(-std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::min(),
            -std::numeric_limits<float>::max(), 0.0f)
    };
    const float2 coordinates[] = {
        float2(0.4664843380f, 0.9199014902f), // Regression: weighted-sum lerp drifted by one ULP.
        float2(0, 0),
        float2(-0.25f, 1.25f)
    };

    for (const auto& color : colors)
    {
        std::array<float4, 6> texels;
        texels.fill(color);
        const Texture2D<float4> texture{ texels.data(), 3, 2 };
        for (float2 uv : coordinates)
            for (uint mode : { Wrap, Mirror, Clamp })
                EXPECT_EQ(Bits(OmmSoftwareSampleLevel(texture, uv, uint2(3, 2), mode, Linear)), Bits(color))
                    << "UV " << uv.x << ',' << uv.y << ", mode " << mode;
    }
}

TEST_F(CpuMathTest, SoftwareSamplingLoadsSpecialValuesWithoutArithmetic)
{
    // Nearest, gather, and border loads preserve signs and NaN payloads exactly.
    constexpr ResultBits colors[] = {
        { 0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u }, // Signed zeros and infinities.
        { 0x7FC12345u, 0xFFC54321u, 0x00000001u, 0x80000001u } // NaN payloads and subnormals.
    };

    for (const auto& color : colors)
    {
        const float4 texel = FromBits(color);
        const Texture2D<float4> texture{ &texel, 1, 1 };
        EXPECT_EQ(Bits(OmmSoftwareSampleLevel(texture, float2(0.5f, 0.5f), uint2(1, 1), Clamp, Nearest)), color);
        for (uint channel = 0; channel < 4; ++channel)
        {
            ResultBits expected;
            expected.fill(color[channel]);
            EXPECT_EQ(Bits(OmmSoftwareGather(texture, float2(0.5f, 0.5f), uint2(1, 1), Clamp, channel)), expected);
            EXPECT_EQ(Bits(OmmSoftwareLoad(texture, int2(-1, 0), uint2(1, 1), Border,
                std::bit_cast<float>(color[channel]))), expected);
        }
    }
}
