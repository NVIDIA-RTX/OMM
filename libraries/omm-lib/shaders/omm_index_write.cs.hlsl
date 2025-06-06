/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/


#include "omm_platform.hlsli"
#include "omm.hlsli"
#include "omm_global_cb.hlsli"
#include "omm_global_samplers.hlsli"
#include "omm_index_write.cs.resources.hlsli"

OMM_DECLARE_LOCAL_CONSTANT_BUFFER
OMM_DECLARE_GLOBAL_CONSTANT_BUFFER
OMM_DECLARE_GLOBAL_SAMPLERS
OMM_DECLARE_INPUT_RESOURCES
OMM_DECLARE_OUTPUT_RESOURCES
OMM_DECLARE_SUBRESOURCES

int Get8BitIndexOrSpecialIndex(int ommIndex)
{
	return 0xFF & ommIndex;
}

int Get16BitIndexOrSpecialIndex(int ommIndex)
{
	return 0xFFFF & ommIndex; // The mask ensures correct treatment of special indices (negative 32 bit values)
}

[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= g_LocalConstants.ThreadCount)
		return;

	const OmmIndexFormat indexFormat = (OmmIndexFormat)g_GlobalConstants.OmmIndexFormat;
	if (indexFormat == OmmIndexFormat::UINT_8) 
	{
		const int4 ommIndexQuad = OMM_SUBRESOURCE_LOAD4(TempOmmIndexBuffer, 4 * (4 * tid.x));
		const int a = Get8BitIndexOrSpecialIndex(ommIndexQuad.x);
		const int b = Get8BitIndexOrSpecialIndex(ommIndexQuad.y);
		const int c = Get8BitIndexOrSpecialIndex(ommIndexQuad.z);
		const int d = Get8BitIndexOrSpecialIndex(ommIndexQuad.w);
		const int ommIndexQuad8Bit = (d << 24u) | (c << 16u) | (b << 8u) | a;
		u_ommIndexBuffer.Store(4 * tid.x, ommIndexQuad8Bit);
	}
	else if (indexFormat == OmmIndexFormat::UINT_16)
	{
		const int2 ommIndexPair = OMM_SUBRESOURCE_LOAD2(TempOmmIndexBuffer, 4 * (2 * tid.x));
		const int a = Get16BitIndexOrSpecialIndex(ommIndexPair.x);
		const int b = Get16BitIndexOrSpecialIndex(ommIndexPair.y);
		const int ommIndexPair16Bit = (b << 16u) | a;
		u_ommIndexBuffer.Store(4 * tid.x, ommIndexPair16Bit);
	}
	else
	{
		const uint ommIndex = OMM_SUBRESOURCE_LOAD(TempOmmIndexBuffer, 4 * tid.x);
		u_ommIndexBuffer.Store(4 * tid.x, ommIndex);
	}
}