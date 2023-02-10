/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>
#include "util/omm.h"
#include "util/image.h"

#include <omm.h>
#include <shared/bird.h>
#include <shared/cpu_raster.h>

#include <algorithm>
#include <random>

namespace {

	class BakeIndexing : public ::testing::Test {
	protected:
		void SetUp() override {
			EXPECT_EQ(omm::CreateBaker({ .type = omm::BakerType::CPU }, &_baker), omm::Result::SUCCESS);
		}

		void TearDown() override {
			for (omm::Cpu::Texture tex : _textures) {
				EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, tex), omm::Result::SUCCESS);
			}

			EXPECT_EQ(omm::DestroyBaker(_baker), omm::Result::SUCCESS);
		}

		omm::Cpu::Texture CreateTexture(const omm::Cpu::TextureDesc& desc) {
			omm::Cpu::Texture tex = 0;
			EXPECT_EQ(omm::Cpu::CreateTexture(_baker, desc, &tex), omm::Result::SUCCESS);
			_textures.push_back(tex);
			return tex;
		}

		void Bake(uint32_t triangleCount, bool force32bitIndexFormat, omm::IndexFormat expectedOutput, omm::Result expectedResult) {

			omm::Cpu::Texture tex_04 = 0;
			{
				vmtest::TextureFP32 texture(1024, 1024, 1, [](int i, int j, int w, int h, int mip) {
					if ((i) % 2 != (j) % 2)
						return 0.f;
					else
						return 1.f;
					});

				tex_04 = CreateTexture(texture.GetDesc());
			}

			uint32_t seed = 32;
			std::default_random_engine eng(seed);
			std::uniform_real_distribution<float> distr(0.f, 1.f);

			uint32_t numIdx = triangleCount * 3;
			std::vector<uint32_t> indices(numIdx);
			std::vector<float2> texCoords(numIdx);
			for (uint32_t i = 0; i < numIdx / 3; ++i) {
				for (uint32_t j = 0; j < 3; ++j) {
					indices[3 * i + j] = 3 * i + j;
					texCoords[3 * i + j] = float2(distr(eng), distr(eng));
				}
			}

			omm::Cpu::BakeInputDesc desc;
			desc.texture = tex_04;
			desc.alphaMode = omm::AlphaMode::Test;
			desc.runtimeSamplerDesc.addressingMode = omm::TextureAddressMode::Clamp;
			desc.runtimeSamplerDesc.filter = omm::TextureFilterMode::Nearest;
			desc.indexFormat = omm::IndexFormat::I32_UINT;
			desc.indexBuffer = indices.data();
			desc.texCoords = texCoords.data();
			desc.texCoordFormat = omm::TexCoordFormat::UV32_FLOAT;
			desc.indexCount = (uint32_t)indices.size();
			desc.maxSubdivisionLevel = 4;
			desc.alphaCutoff = 0.3f;
			desc.bakeFlags = (omm::Cpu::BakeFlags)(
				(uint32_t)omm::Cpu::BakeFlags::EnableInternalThreads |
				(uint32_t)omm::Cpu::BakeFlags::DisableSpecialIndices |
				(uint32_t)omm::Cpu::BakeFlags::DisableDuplicateDetection);

			if (force32bitIndexFormat)
				desc.bakeFlags = (omm::Cpu::BakeFlags)((uint32_t)omm::Cpu::BakeFlags::Force32BitIndices | (uint32_t)desc.bakeFlags);

			desc.dynamicSubdivisionScale = 0.f;
			omm::Cpu::BakeResult res = 0;

			EXPECT_EQ(omm::Cpu::Bake(_baker, desc, &res), expectedResult);
			if (expectedResult != omm::Result::SUCCESS)
				return;
			EXPECT_NE(res, 0);

			const omm::Cpu::BakeResultDesc* resDesc = nullptr;
			EXPECT_EQ(omm::Cpu::GetBakeResultDesc(res, &resDesc), omm::Result::SUCCESS);
			EXPECT_NE(resDesc, nullptr);

			EXPECT_EQ(resDesc->indexFormat, expectedOutput);

			EXPECT_EQ(resDesc->indexCount, triangleCount);

			EXPECT_EQ(omm::Cpu::DestroyBakeResult(res), omm::Result::SUCCESS);

		}

		std::vector< omm::Cpu::Texture> _textures;
		omm::Baker _baker = 0;
	};

	// Any
	TEST_F(BakeIndexing, TriangleCount_1) {
		Bake(1, false /*force32bitIndices*/,
			omm::IndexFormat::I16_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_32766) {
		Bake(32766, false /*force32bitIndices*/,
			omm::IndexFormat::I16_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_32767) {
		Bake(32767, false /*force32bitIndices*/,
			omm::IndexFormat::I16_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_32768) {
		Bake(32768, false /*force32bitIndices*/,
			omm::IndexFormat::I32_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_65536) {
		Bake(65536, false /*force32bitIndices*/,
			omm::IndexFormat::I32_UINT, omm::Result::SUCCESS);
	}

	// forceI32
	TEST_F(BakeIndexing, TriangleCount_1_ForceI32) {
		Bake(1, true /*force32bitIndices*/,
			omm::IndexFormat::I32_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_32766_ForceI32) {
		Bake(32766, true /*force32bitIndices*/,
			omm::IndexFormat::I32_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_32767_ForceI32) {
		Bake(32767, true /*force32bitIndices*/,
			omm::IndexFormat::I32_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_32768_ForceI32) {
		Bake(32768, true /*force32bitIndices*/,
			omm::IndexFormat::I32_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_65536_ForceI32) {
		Bake(65536, false /*force32bitIndices*/, 
			omm::IndexFormat::I32_UINT, omm::Result::SUCCESS);
	}

	// PreferI16
	TEST_F(BakeIndexing, TriangleCount_1_ForceI16) {
		Bake(1, false /*force32bitIndices*/,
			omm::IndexFormat::I16_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_32766_ForceI16) {
		Bake(32766, false /*force32bitIndices*/,
			omm::IndexFormat::I16_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_32767_ForceI16) {
		Bake(32767, false /*force32bitIndices*/,
			omm::IndexFormat::I16_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_32768_ForceI16) {
		Bake(32768, false /*force32bitIndices*/,
			omm::IndexFormat::I32_UINT, omm::Result::SUCCESS);
	}

	TEST_F(BakeIndexing, TriangleCount_65536_ForceI16) {
		Bake(65536, false /*force32bitIndices*/,
			omm::IndexFormat::I32_UINT, omm::Result::SUCCESS);
	}

}  // namespace