/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>
#include "util/texture.h"
#include "texture_impl.h"
#include <omm.h>
#include <array>
#include <vector>

namespace {

	const char* GetName(omm::TextureAddressMode mode) {
		switch (mode) {
		case omm::TextureAddressMode::Wrap:
			return "Wrap";
		case omm::TextureAddressMode::Mirror:
			return "Mirror";
		case omm::TextureAddressMode::Clamp:
			return "Clamp";
		case omm::TextureAddressMode::Border:
			return "Border";
		case omm::TextureAddressMode::MirrorOnce:
			return "MirrorOnce";
		default:
			return "Unknown";
		}
	}

	void TexCoordTest(omm::TextureAddressMode mode, int2 texCoord, int2 size, int2 expectedValue) {
		int2 modifiedTex = omm::GetTexCoord(mode, omm::isPow2(size.x) && omm::isPow2(size.y), texCoord, size, omm::ctz(size));
		EXPECT_EQ(modifiedTex, expectedValue) << "Input:[" << texCoord.x << "," << texCoord.y << "],Expected:[" << expectedValue.x << "," << expectedValue.y
			<< "],Was:[" << modifiedTex.x << "," << modifiedTex.y << "] Mode:" << GetName(mode);
	}

	TEST(GetTexCoord, Wrap) {
		//													[Mode, TexCoord, Size, Expected]
		TexCoordTest(omm::TextureAddressMode::Wrap, { 512, 512 },	{ 1024, 1024 }, { 512, 512 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 0, 512 },		{ 1024, 1024 }, { 0, 512 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 0, 0 },		{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -1, -1 },		{ 1024, 1024 }, { 1023, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -1024, -1 },	{ 1024, 1024 }, { 0, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -2048, -1 },	{ 1024, 1024 }, { 0, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 1024, 1024 },	{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 2048, 1024 },	{ 1024, 1024 }, { 0, 0 });

		TexCoordTest(omm::TextureAddressMode::Wrap, { 512, 512 },    { 512, 1024 },  { 0, 512 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 0, 512 },		{ 512, 1024 },  { 0, 512 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 0, 0 },		{ 512, 1024 },  { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -1, -1 },		{ 512, 1024 },  { 511, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -1024, -1 },   { 512, 1024 },  { 0, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -2048, -1 },   { 512, 1024 },  { 0, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 1024, 1024 },  { 512, 1024 },  { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 2048, 1024 },  { 512, 1024 },  { 0, 0 });
	}

	TEST(GetTexCoord, Mirror) {
		//												 [Mode, TexCoord, Size, Expected]
		// Positive X
		TexCoordTest(omm::TextureAddressMode::Mirror, { 0, 4 },  { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 1, 4 },  { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 2, 4 },  { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 3, 4 },  { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 4 },  { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 5, 4 },  { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 6, 4 },  { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 7, 4 },  { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 8, 4 },  { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 9, 4 },  { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 10, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 11, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 12, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 13, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 14, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 15, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 16, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 17, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 18, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 19, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 20, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 21, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 22, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 23, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 24, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 25, 4 }, { 8, 8 }, { 6, 4 });

		// Negative X
		TexCoordTest(omm::TextureAddressMode::Mirror, { -0, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, {  0, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -1, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -2, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -3, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -4, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -5, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -6, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -7, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -8, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -9, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -10, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -11, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -12, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -13, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -14, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -15, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -16, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -17, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -18, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -19, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -20, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -21, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -22, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -23, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -24, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -25, 4 }, { 8, 8 }, { 7, 4 });

		// Positive Y
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  0 }, { 8, 8 }, { 4, 0, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  1 }, { 8, 8 }, { 4, 1, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  2 }, { 8, 8 }, { 4, 2, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  3 }, { 8, 8 }, { 4, 3, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  4 }, { 8, 8 }, { 4, 4, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  5 }, { 8, 8 }, { 4, 5, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  6 }, { 8, 8 }, { 4, 6, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  7 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  8 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  9 }, { 8, 8 }, { 4, 6, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 10 }, { 8, 8 }, { 4, 5, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 11 }, { 8, 8 }, { 4, 4, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 12 }, { 8, 8 }, { 4, 3, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 13 }, { 8, 8 }, { 4, 2, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 14 }, { 8, 8 }, { 4, 1, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 15 }, { 8, 8 }, { 4, 0, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 16 }, { 8, 8 }, { 4, 0, });

		// Positive X,Y
		TexCoordTest(omm::TextureAddressMode::Mirror, { 8, 8 },   { 8, 8 }, { 7, 7 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 16, 16 }, { 8, 8 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 32, 32 }, { 8, 8 }, { 0, 0 });
	}

	TEST(GetTexCoord, Clamp) {
		//												[Mode, TexCoord, Size, Expected]
		TexCoordTest(omm::TextureAddressMode::Clamp, { 512, 512 },	{ 1024, 1024 }, { 512, 512 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { 0, 512 },		{ 1024, 1024 }, { 0, 512 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { 0, 0 },		{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { -1, -1 },		{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { -1024, -1 },	{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { -2048, -1 },	{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { 1024, 1024 }, { 1024, 1024 }, { 1023, 1023 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { 2048, 1024 }, { 1024, 1024 }, { 1023, 1023 });
	}

	TEST(GetTexCoord, Border) {
		//												[Mode, TexCoord, Size, Expected]
		TexCoordTest(omm::TextureAddressMode::Border, { 512, 512 }, { 512, 1024 }, int2{ omm::kTexCoordBorder, 512 });
		TexCoordTest(omm::TextureAddressMode::Border, { 0, 512 }, { 512, 1024 }, { 0, 512 });
		TexCoordTest(omm::TextureAddressMode::Border, { 0, 0 }, { 512, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Border, { -1, -1 }, { 512, 1024 }, omm::kTexCoordBorder2);
		TexCoordTest(omm::TextureAddressMode::Border, { 0, -1 }, { 512, 1024 }, int2{ 0, omm::kTexCoordBorder });
		TexCoordTest(omm::TextureAddressMode::Border, { -1024, -1 }, { 512, 1024 }, omm::kTexCoordBorder2);
		TexCoordTest(omm::TextureAddressMode::Border, { -2048, -1 }, { 512, 1024 }, omm::kTexCoordBorder2);
		TexCoordTest(omm::TextureAddressMode::Border, { 1024, 1024 }, { 512, 1024 }, omm::kTexCoordBorder2);
		TexCoordTest(omm::TextureAddressMode::Border, { 2048, 1024 }, { 512, 1024 }, omm::kTexCoordBorder2);
	}

	TEST(GetTexCoord, MirrorOnce) {

		//													[Mode, TexCoord, Size, Expected]
		// Positive X
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 0, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 1, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 2, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 3, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 5, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 6, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 7, 4 }, { 8, 8 }, { 7, 4 });
		// no more mirroring -> clamp
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 8, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 9, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 10, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 11, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 12, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 13, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 14, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 15, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 16, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 17, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 18, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 19, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 20, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 21, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 22, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 23, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 24, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 25, 4 }, { 8, 8 }, { 7, 4 });

		// Negative X
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -0, 4 }, { 8, 8 }, { 0, 4 });
		// First mirror...
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 0, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -1, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -2, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -3, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -4, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -5, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -6, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -7, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -8, 4 }, { 8, 8 }, { 7, 4 });
		// no more mirroring -> clamp
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -9, 4 }, { 8, 8 },  { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -10, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -11, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -12, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -13, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -14, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -15, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -16, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -17, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -18, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -19, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -20, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -21, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -22, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -23, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -24, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -25, 4 }, { 8, 8 }, { 7, 4 });

		// Positive Y
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  0 }, { 8, 8 }, { 4, 0, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  1 }, { 8, 8 }, { 4, 1, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  2 }, { 8, 8 }, { 4, 2, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  3 }, { 8, 8 }, { 4, 3, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  4 }, { 8, 8 }, { 4, 4, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  5 }, { 8, 8 }, { 4, 5, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  6 }, { 8, 8 }, { 4, 6, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  7 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  8 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  9 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 10 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 11 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 12 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 13 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 14 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 15 }, { 8, 8 }, { 4, 7, });
		// no more mirroring -> clamp
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 16 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 17 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 18 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 19 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 20 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 21 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 22 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 23 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 24 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 25 }, { 8, 8 }, { 4, 7 });

		// Positive X,Y
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 8, 8 },   { 8, 8 },  { 7, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 16, 16 }, { 8, 8 },  { 7, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 32, 32 }, { 8, 8 },  { 7, 7 });
	}

	template<ommCpuTextureFormat Format, omm::TilingMode Tiling>
	struct TextureSamplingConfig {
		static constexpr ommCpuTextureFormat format = Format;
		static constexpr omm::TilingMode tiling = Tiling;
	};

	template<class Config>
	class TextureSampling : public ::testing::Test {
	protected:
		void SetUp() override {
			ommBakerCreationDesc bakerDesc = ommBakerCreationDescDefault();
			bakerDesc.type = ommBakerType_CPU;
			ASSERT_EQ(ommCreateBaker(&bakerDesc, &_baker), ommResult_SUCCESS);

			std::array<ommCpuTextureMipDesc, 3> mips;
			for (int mip = 0; mip < 3; ++mip) {
				const int2 size = _sizes[mip];
				for (int y = 0; y < size.y; ++y) {
					for (int x = 0; x < size.x; ++x) {
						_data8[mip].push_back(uint8_t((x * 37 + y * 83 + mip * 17) % 256));
						_data32[mip].push_back(float((x * 17 + y * 31 + mip * 11) % 64) / 64.f);
					}
				}
				mips[mip] = ommCpuTextureMipDescDefault();
				mips[mip].width = size.x;
				mips[mip].height = size.y;
				if constexpr (Config::format == ommCpuTextureFormat_FP32)
					mips[mip].textureData = _data32[mip].data();
				else
					mips[mip].textureData = _data8[mip].data();
			}

			ommCpuTextureDesc textureDesc = ommCpuTextureDescDefault();
			textureDesc.format = Config::format;
			textureDesc.flags = Config::tiling == omm::TilingMode::Linear ? ommCpuTextureFlags_DisableZOrder : ommCpuTextureFlags_None;
			textureDesc.mips = mips.data();
			textureDesc.mipCount = uint32_t(mips.size());
			ASSERT_EQ(ommCpuCreateTexture(_baker, &textureDesc, &_texture), ommResult_SUCCESS);
		}

		void TearDown() override {
			if (_texture)
				EXPECT_EQ(ommCpuDestroyTexture(_baker, _texture), ommResult_SUCCESS);
			if (_baker)
				EXPECT_EQ(ommDestroyBaker(_baker), ommResult_SUCCESS);
		}

		const omm::TextureImpl& Texture() const {
			return *omm::GetHandleImpl<omm::TextureImpl>(_texture);
		}

		float ReferenceLoad(const int2& coord, int mip, float borderAlpha) const {
			if (coord.x == omm::kTexCoordBorder || coord.y == omm::kTexCoordBorder)
				return borderAlpha;
			const size_t index = size_t(coord.y) * _sizes[mip].x + coord.x;
			if constexpr (Config::format == ommCpuTextureFormat_FP32)
				return _data32[mip][index];
			else
				return float(_data8[mip][index]) * (1.f / 255.f);
		}

		template<ommTextureAddressMode Mode, bool IsPow2>
		void CheckBilinear(int mip) const {
			const int2 size = _sizes[mip];
			const float uv[] = { -1.125f, -0.25f, -0.0625f, 0.f, 0.0625f, 0.25f, 0.5f, 0.875f, 0.9375f, 1.f, 1.0625f, 1.375f, 2.125f };
			for (float v : uv) {
				for (float u : uv) {
					for (float borderAlpha : { 0.f, 0.375f, 1.f }) {
						SCOPED_TRACE(::testing::Message() << "mip=" << mip << " mode=" << int(Mode) << " uv=" << u << "," << v << " border=" << borderAlpha);
						const float2 pixel = float2(u, v) * float2(size) - 0.5f;
						const int2 base = int2(glm::floor(pixel));
						const float2 w = pixel - float2(base);
						auto sample = [&](int x, int y) {
							return ReferenceLoad(omm::GetTexCoord(Mode, IsPow2, base + int2(x, y), size, omm::ctz(size)), mip, borderAlpha);
						};
						const float expected = (sample(0, 0) * (1.f - w.x) + sample(1, 0) * w.x) * (1.f - w.y)
							+ (sample(0, 1) * (1.f - w.x) + sample(1, 1) * w.x) * w.y;
						const float actual = Texture().template Bilinear<Config::format, Config::tiling, Mode, IsPow2>(float2(u, v), mip, borderAlpha);
						EXPECT_NEAR(actual, expected, 1e-6f);
					}
				}
			}
		}

		const std::array<int2, 3> _sizes = { int2(8, 4), int2(5, 3), int2(1, 1) };
		std::array<std::vector<uint8_t>, 3> _data8;
		std::array<std::vector<float>, 3> _data32;
		ommBaker _baker = nullptr;
		ommCpuTexture _texture = nullptr;
	};

	using TextureSamplingTypes = ::testing::Types<
		TextureSamplingConfig<ommCpuTextureFormat_FP32, omm::TilingMode::Linear>,
		TextureSamplingConfig<ommCpuTextureFormat_FP32, omm::TilingMode::MortonZ>,
		TextureSamplingConfig<ommCpuTextureFormat_UNORM8, omm::TilingMode::Linear>,
		TextureSamplingConfig<ommCpuTextureFormat_UNORM8, omm::TilingMode::MortonZ>>;
	TYPED_TEST_SUITE(TextureSampling, TextureSamplingTypes);

	TYPED_TEST(TextureSampling, GatherMatchesSourceTexels) {
		for (int mip = 0; mip < 3; ++mip) {
			const int2 size = this->_sizes[mip];
			const bool isPow2 = omm::isPow2(size.x) && omm::isPow2(size.y);
			for (int mode = 0; mode < ommTextureAddressMode_MAX_NUM; ++mode) {
				for (int y = -size.y - 2; y <= size.y + 1; ++y) {
					for (int x = -size.x - 2; x <= size.x + 1; ++x) {
						int2 coords[omm::TexelOffset::MAX_NUM];
						omm::GatherTexCoord4(ommTextureAddressMode(mode), isPow2, int2(x, y), size, omm::ctz(size), coords);
						for (float borderAlpha : { 0.f, 0.375f, 1.f }) {
							SCOPED_TRACE(::testing::Message() << "mip=" << mip << " mode=" << mode << " cell=" << x << "," << y << " border=" << borderAlpha);
							const float4 actual = mode == ommTextureAddressMode_Border
								? this->Texture().template Gather<TypeParam::format, TypeParam::tiling, true>(coords[omm::I0x0], coords[omm::I1x1], mip, borderAlpha)
								: this->Texture().template Gather<TypeParam::format, TypeParam::tiling>(coords[omm::I0x0], coords[omm::I1x1], mip, borderAlpha);
							EXPECT_FLOAT_EQ(actual.x, this->ReferenceLoad(coords[omm::I0x0], mip, borderAlpha));
							EXPECT_FLOAT_EQ(actual.y, this->ReferenceLoad(coords[omm::I0x1], mip, borderAlpha));
							EXPECT_FLOAT_EQ(actual.z, this->ReferenceLoad(coords[omm::I1x1], mip, borderAlpha));
							EXPECT_FLOAT_EQ(actual.w, this->ReferenceLoad(coords[omm::I1x0], mip, borderAlpha));
						}
					}
				}
			}
		}
	}

	TYPED_TEST(TextureSampling, BilinearMatchesScalarInterpolation) {
		for (int mip = 0; mip < 3; ++mip) {
			const int2 size = this->_sizes[mip];
			if (omm::isPow2(size.x) && omm::isPow2(size.y)) {
				this->template CheckBilinear<ommTextureAddressMode_Wrap, true>(mip);
				this->template CheckBilinear<ommTextureAddressMode_Mirror, true>(mip);
				this->template CheckBilinear<ommTextureAddressMode_Clamp, true>(mip);
				this->template CheckBilinear<ommTextureAddressMode_Border, true>(mip);
				this->template CheckBilinear<ommTextureAddressMode_MirrorOnce, true>(mip);
			}
			else {
				this->template CheckBilinear<ommTextureAddressMode_Wrap, false>(mip);
				this->template CheckBilinear<ommTextureAddressMode_Mirror, false>(mip);
				this->template CheckBilinear<ommTextureAddressMode_Clamp, false>(mip);
				this->template CheckBilinear<ommTextureAddressMode_Border, false>(mip);
				this->template CheckBilinear<ommTextureAddressMode_MirrorOnce, false>(mip);
			}
		}
	}
}
