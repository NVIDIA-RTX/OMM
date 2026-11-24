/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef OMM_INDEX_BUFFER_HLSLI
#define OMM_INDEX_BUFFER_HLSLI

// Custom shader integrations may set this before including OMM headers.
// The SDK host code and shaders must use the same setting.
#ifndef OMM_USE_RAW_INDEX_BUFFER
#define OMM_USE_RAW_INDEX_BUFFER 0
#endif

#if OMM_USE_RAW_INDEX_BUFFER
#define OMM_INDEX_BUFFER_TYPE ByteAddressBuffer
#else
#define OMM_INDEX_BUFFER_TYPE Buffer<uint>
#endif

#endif
