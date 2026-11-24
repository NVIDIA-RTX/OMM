/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef OMM_SOFTWARE_DIVIDE_HLSLI
#define OMM_SOFTWARE_DIVIDE_HLSLI

// Shared integer implementation, also called directly by CPU unit tests.
// Restoring division reference: Structure of Computer Systems, Section 3.3.1:
// https://users.utcluj.ro/~baruch/en/media/book_ssce/SSCE-Basic-Division.pdf
// The shared utility explains normalization and exact-remainder rounding.
#include "../src/util/omm_software_math.h"

#endif // OMM_SOFTWARE_DIVIDE_HLSLI
