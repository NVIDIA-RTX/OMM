/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include "shader_cb.h"
#include "util.hlsli"


struct OmmDesc
{
    uint32_t offset;
    uint16_t subdivisionLevel;
    uint16_t format;
};

SamplerState s_SamplerLinear : register(s0);
SamplerState s_SamplerPoint : register(s1);
RasterizerOrderedTexture2D<float4> u_ReadbackTexture : register(u0);
Texture2D t_Texture : register(t0);
Texture2D t_TextureMin : register(t1);
Texture2D t_TextureMax : register(t2);
Buffer<int> t_OmmIndexBuffer : register(t3);
StructuredBuffer<OmmDesc> t_OmmDesc : register(t4);
ByteAddressBuffer t_OmmArrayData : register(t5);

cbuffer c_Constants : register(b0)
{
    Constants g_constants;
};

void main_vs(
	in float2 i_texCoord : SV_Position,
	out float4 o_pos : SV_Position,
	out float2 o_texCoord : TEXCOORD0
)
{
    o_texCoord = i_texCoord;
    float2 vert = 2 * i_texCoord - 1.0;
    
    vert += g_constants.offset;
    vert *= g_constants.zoom;
    vert *= g_constants.aspectRatio;

	o_pos = float4(vert, 0, 1);
}

static inline uint prefixEor2(uint x)
{
    x ^= (x >> 1) & 0x7fff7fff;
    x ^= (x >> 2) & 0x3fff3fff;
    x ^= (x >> 4) & 0x0fff0fff;
    x ^= (x >> 8) & 0x00ff00ff;
    return x;
}

// Interleave 16 even bits from x with 16 odd bits from y
static inline uint interleaveBits2(uint x, uint y)
{
    x = (x & 0xffff) | (y << 16);
    x = ((x >> 8) & 0x0000ff00) | ((x << 8) & 0x00ff0000) | (x & 0xff0000ff);
    x = ((x >> 4) & 0x00f000f0) | ((x << 4) & 0x0f000f00) | (x & 0xf00ff00f);
    x = ((x >> 2) & 0x0c0c0c0c) | ((x << 2) & 0x30303030) | (x & 0xc3c3c3c3);
    x = ((x >> 1) & 0x22222222) | ((x << 1) & 0x44444444) | (x & 0x99999999);

    return x;
}

static uint dbary2index(uint u, uint v, uint w, uint level)
{
    const uint coordMask = ((1U << level) - 1);

    uint b0 = ~(u ^ w) & coordMask;
    uint t = (u ^ v) & b0; //  (equiv: (~u & v & ~w) | (u & ~v & w))
    uint c = (((u & v & w) | (~u & ~v & ~w)) & coordMask) << 16;
    uint f = prefixEor2(t | c) ^ u;
    uint b1 = (f & ~b0) | t; // equiv: (~u & v & ~w) | (u & ~v & w) | (f0 & u & ~w) | (f0 & ~u & w))

    return interleaveBits2(b0, b1); // 13 instructions
}

static uint bary2index(float2 bc, uint level, out bool isUpright)
{
    float numSteps = float(1u << level);
    uint iu = uint(numSteps * bc.x);
    uint iv = uint(numSteps * bc.y);
    uint iw = uint(numSteps * (1.f - bc.x - bc.y));
    isUpright = (iu & 1) ^ (iv & 1) ^ (iw & 1);
    return dbary2index(iu, iv, iw, level);
}

float3 MicroStateColor(int state)
{
    if (g_constants.colorizeStates)
    {
        if (state == 0)
            return float3(0, 0, 1.f);
        if (state == 1)
            return float3(0, 1, 0.f);
        if (state == 2)
            return float3(1.f, 0, 1.f);
    //if (state == 3)

        return float3(1.f, 1.f, 0.f);
    }
    else
    {
        if (state == 0)
            return float3(0, 0, 0.f);
        if (state == 1)
            return float3(1, 1, 1.f);
        if (state == 2)
            return float3(1.f, 0, 0.f);
    //if (state == 3)
        return float3(0.f, 0.f, 0.f);
    }
}

void main_ps(
	in float4 i_pos : SV_Position,
	in float2 i_texCoord : TEXCOORD0,
	in uint i_primitiveId : SV_PrimitiveID,
    in float3 bc : SV_Barycentrics,
    in bool isFrontFace : SV_IsFrontFace,
	out float4 o_color : SV_Target0
)
{
    int ommIndex = t_OmmIndexBuffer[i_primitiveId + g_constants.primitiveOffset];

    if (g_constants.ommIndexIsolate >= 0 && ommIndex != g_constants.ommIndexIsolate)
    {
        discard;
    }
    
    u_ReadbackTexture[i_pos.xy].w = asfloat(ommIndex + 5);
    
    float highlight = g_constants.ommIndexHighlightEnable ? 0.5f : 1.f;
    if (g_constants.ommIndexHighlightEnable &&
        ommIndex == g_constants.ommIndexHighlight)
    {
        highlight = 1.0f;
    }
    
    if (g_constants.mode == 1) // wireframe
    {
        o_color = float4(1, 0, 0, 1.0);
        return;
    }
    
    uint primitiveIndex = i_primitiveId + g_constants.primitiveOffset;
    
    uint ommIndexBufferSize = 0;
    t_OmmIndexBuffer.GetDimensions(ommIndexBufferSize);
    
    if (primitiveIndex >= ommIndexBufferSize)
    {
        discard;
    }
    
    const bool isIntersection = IsOverIntersectionLine(t_Texture, t_TextureMin, t_TextureMax, s_SamplerLinear, g_constants.invTexSize, g_constants.alphaCutoff, i_texCoord);
    
    float3 color = float3(0, 0, 0);
    if (g_constants.drawAlphaContour && isIntersection)
    {
        o_color = float4(kContourLineColor, 1.0);
        return;
    }
    
    if (ommIndex < 0)
    {
        o_color = highlight * float4(MicroStateColor(-(ommIndex + 1)), 0.5);
        return;
    }

    OmmDesc ommDesc = t_OmmDesc[ommIndex];
    const bool is2State = ommDesc.format == 1;
    
    bool isUpright;
    const uint microIndex = bary2index(bc.yz, ommDesc.subdivisionLevel, isUpright);
    const uint statesPerDW = is2State ? 32 : 16;
    const uint startOffset = ommDesc.offset;
    const uint offsetDW = startOffset + 4 * (microIndex / statesPerDW);
    uint stateDW = t_OmmArrayData.Load(offsetDW);
    const uint bitOffset = (is2State ? 1 : 2) * (microIndex % statesPerDW);
    const uint state = (stateDW >> bitOffset) & (is2State ? 0x1u : 0x3u);
    
    float3 clr = MicroStateColor(state);
    clr *= 0.5;
    if (isUpright)
    {
        clr *= 0.5f;
    }
    
    clr *= highlight;
    
    const float alphaLerp = t_Texture.Sample(s_SamplerLinear, i_texCoord).r;

    {
        color = 0.01 * alphaLerp.xxx;
    }

    o_color = float4(clr.xyz + 0.5 * color, 1.0);
}
