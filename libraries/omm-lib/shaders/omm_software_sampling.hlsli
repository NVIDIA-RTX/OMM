/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef OMM_SOFTWARE_SAMPLING_HLSLI
#define OMM_SOFTWARE_SAMPLING_HLSLI

// Address modes match ommTextureAddressMode: Wrap, Mirror, Clamp, Border.
// Coordinates must be finite with texel indices representable as signed int32,
// as required by the compute rasterizer's integer bounding box as well.
int OmmSoftwareAddressTexel(int coordinate, int size, uint addressMode)
{
    if (addressMode == 0u)
    {
        const int wrapped = coordinate % size;
        return wrapped < 0 ? wrapped + size : wrapped;
    }
    if (addressMode == 1u)
    {
        const int period = 2 * size;
        int wrapped = coordinate % period;
        if (wrapped < 0)
            wrapped += period;
        return wrapped < size ? wrapped : period - 1 - wrapped;
    }
    return clamp(coordinate, 0, size - 1);
}

float4 OmmSoftwareLoad(Texture2D<float4> texture, int2 texel, uint2 textureSize,
    uint addressMode, float borderAlpha)
{
    if (addressMode == 3u && (any(texel < 0) || any(texel >= int2(textureSize))))
        return float4(borderAlpha, borderAlpha, borderAlpha, borderAlpha);
    const int2 addressed = int2(
        OmmSoftwareAddressTexel(texel.x, int(textureSize.x), addressMode),
        OmmSoftwareAddressTexel(texel.y, int(textureSize.y), addressMode));
    return texture.Load(int3(addressed, 0));
}

float4 OmmSoftwareLerp(float4 a, float4 b, float weight)
{
    // Difference form preserves finite constant colors exactly.
    const precise float4 difference = b - a;
    const precise float4 scaled = difference * weight;
    const precise float4 result = a + scaled;
    return result;
}

float4 OmmSoftwareSampleLevel(Texture2D<float4> texture, float2 uv, uint2 textureSize,
    uint addressMode, uint filterMode, float borderAlpha = 0.0f)
{
    const precise float2 scaled = uv * float2(textureSize);
    if (filterMode == 0u) // Nearest
        return OmmSoftwareLoad(texture, int2(floor(scaled)), textureSize, addressMode, borderAlpha);

    // Full float32 weights, with no hardware subtexel snapping. Spell out each
    // operation so precise compilation fixes the interpolation evaluation order.
    const precise float2 position = scaled - 0.5f;
    const precise float2 base = floor(position);
    const int2 texel = int2(base);
    const precise float2 weight = position - base;
    const float4 a = OmmSoftwareLoad(texture, texel, textureSize, addressMode, borderAlpha);
    const float4 b = OmmSoftwareLoad(texture, texel + int2(1, 0), textureSize, addressMode, borderAlpha);
    const float4 c = OmmSoftwareLoad(texture, texel + int2(0, 1), textureSize, addressMode, borderAlpha);
    const float4 d = OmmSoftwareLoad(texture, texel + int2(1, 1), textureSize, addressMode, borderAlpha);
    const precise float4 row0 = OmmSoftwareLerp(a, b, weight.x);
    const precise float4 row1 = OmmSoftwareLerp(c, d, weight.x);
    return OmmSoftwareLerp(row0, row1, weight.y);
}

float4 OmmSoftwareGather(Texture2D<float4> texture, float2 uv, uint2 textureSize,
    uint addressMode, uint channel, float borderAlpha = 0.0f)
{
    const precise float2 scaled = uv * float2(textureSize);
    const precise float2 position = scaled - 0.5f;
    const int2 texel = int2(floor(position));
    // Match Gather{Red,Green,Blue,Alpha} ordering: (-,+), (+,+), (+,-), (-,-).
    return float4(
        OmmSoftwareLoad(texture, texel + int2(0, 1), textureSize, addressMode, borderAlpha)[channel],
        OmmSoftwareLoad(texture, texel + int2(1, 1), textureSize, addressMode, borderAlpha)[channel],
        OmmSoftwareLoad(texture, texel + int2(1, 0), textureSize, addressMode, borderAlpha)[channel],
        OmmSoftwareLoad(texture, texel, textureSize, addressMode, borderAlpha)[channel]);
}

#endif // OMM_SOFTWARE_SAMPLING_HLSLI
