/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Only the scalar/vector and texture operations used by the software sampling
// shader. The shader itself supplies all addressing and filtering algorithms.
namespace hlsl_cpu
{
    using uint = uint32_t;
    static_assert(sizeof(int) == sizeof(int32_t));

    struct uint2
    {
        uint x, y;
        uint2(uint x, uint y) : x(x), y(y) { }
    };

    struct float2
    {
        float x, y;
        float2(float x, float y) : x(x), y(y) { }
        explicit float2(uint2 v) : x(float(v.x)), y(float(v.y)) { }
    };

    struct int2
    {
        int x, y;
        int2(int x, int y) : x(x), y(y) { }
        explicit int2(uint2 v) : x(int(v.x)), y(int(v.y)) { }
        explicit int2(float2 v) : x(int(v.x)), y(int(v.y)) { }
    };

    struct int3
    {
        int x, y, z;
        int3(int2 xy, int z) : x(xy.x), y(xy.y), z(z) { }
    };

    struct bool2 { bool x, y; };

    struct float4
    {
        std::array<float, 4> values{};
        float4() = default;
        float4(float x, float y, float z, float w) : values{ x, y, z, w } { }
        float& operator[](uint channel) { return values.at(channel); }
        float operator[](uint channel) const { return values.at(channel); }
    };

    inline float2 operator*(float2 a, float2 b) { return { a.x * b.x, a.y * b.y }; }
    inline float2 operator-(float2 a, float2 b) { return { a.x - b.x, a.y - b.y }; }
    inline float2 operator-(float2 a, float b) { return { a.x - b, a.y - b }; }
    inline int2 operator+(int2 a, int2 b) { return { a.x + b.x, a.y + b.y }; }
    inline bool2 operator<(int2 a, int b) { return { a.x < b, a.y < b }; }
    inline bool2 operator>=(int2 a, int2 b) { return { a.x >= b.x, a.y >= b.y }; }
    inline bool any(bool2 a) { return a.x || a.y; }
    inline int clamp(int value, int lower, int upper) { return std::clamp(value, lower, upper); }
    inline float2 floor(float2 value) { return { std::floor(value.x), std::floor(value.y) }; }

    inline float4 operator+(float4 a, float4 b)
    {
        return { a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3] };
    }
    inline float4 operator-(float4 a, float4 b)
    {
        return { a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3] };
    }
    inline float4 operator*(float4 a, float b)
    {
        return { a[0] * b, a[1] * b, a[2] * b, a[3] * b };
    }

    template<typename T>
    struct Texture2D
    {
        const T* texels;
        uint width, height;

        T Load(int3 location) const
        {
            // Invalid shader addressing must fail the CPU test, rather than
            // escaping the texture allocation and hiding the underlying bug.
            if (location.z != 0 || location.x < 0 || location.y < 0 ||
                uint(location.x) >= width || uint(location.y) >= height)
                throw std::out_of_range("software sampler loaded outside its texture");
            return texels[std::size_t(location.y) * width + uint(location.x)];
        }
    };
}
