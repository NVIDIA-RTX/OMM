/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef OMM_SOFTWARE_SQRT_HLSLI
#define OMM_SOFTWARE_SQRT_HLSLI

// Shared integer implementation, also called directly by CPU unit tests.
// Restoring square-root recurrence: Li and Chu, ICCD'97, Section 2.1:
// https://yamin.cis.k.hosei.ac.jp/papers/ICCD97.pdf
// Our binary32 normalization and exact-remainder rounding are explained
// beside OmmSoftwareSqrt in the shared utility.
#include "../src/util/omm_software_math.h"

#endif // OMM_SOFTWARE_SQRT_HLSLI
