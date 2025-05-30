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
#include "util/omm_histogram.h"

#include <stb_image.h>

#include <omm.h>
#include "util/bird.h"

#include <math.h>
#include <cmath>

#include <string>
#include <fstream>
#include <vector>
#include <istream>
#include <iterator>

namespace {

	enum TestSuiteConfig
	{
		Default,
		TextureDisableZOrder,
		Force32BitIndices,
		TextureAsUNORM8,
		AlphaCutoff,
		Serialize,
	};

	struct Options
	{
		omm::Format format = omm::Format::OC1_4_State;
		omm::TextureAddressMode addressingMode = omm::TextureAddressMode::Clamp;
		omm::UnknownStatePromotion unknownStatePromotion = omm::UnknownStatePromotion::Nearest;
		bool mergeSimilar = false;
		uint32_t mipCount = 1;
		bool enableSpecialIndices = true;
		bool oneFile = true;
		bool detailedCutout = false;
		bool monochromeUnknowns = false;
		omm::OpacityState alphaCutoffLE = omm::OpacityState::Transparent;
		omm::OpacityState alphaCutoffGT = omm::OpacityState::Opaque;
		uint64_t maxWorkloadSize = 0xFFFFFFFFFFFFFFFF;
		omm::Result bakeResult = omm::Result::SUCCESS;
		bool forceCorruptedBlob = false;
		bool forceSerializedOutput = false;
		bool serializeCompress = false;
		omm::SpecialIndex unresolvedTriState = omm::SpecialIndex::FullyUnknownOpaque;
		float dynamicSubdivisionScale = 0.f;
	};

	static float StandardCircle(int i, int j, int w, int h, int mip)
	{
		if (i == 0 && j == 0)
			return 0.6f;

		const float r = 0.4f;

		const int2 idx = int2(i, j);
		const float2 uv = float2(idx) / float2((float)w);
		if (glm::length(uv - 0.5f) < r)
			return 0.f;
		return 1.f;
	}

	class OMMBakeTestCPU : public ::testing::TestWithParam<TestSuiteConfig> {
	protected:
		void SetUp() override {
			EXPECT_EQ(omm::CreateBaker({.type = omm::BakerType::CPU }, &_baker), omm::Result::SUCCESS);
		}

		void TearDown() override {
			for (omm::Cpu::Texture tex : _textures) {
				EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, tex), omm::Result::SUCCESS);
			}

			EXPECT_EQ(omm::DestroyBaker(_baker), omm::Result::SUCCESS);
		}

		bool EnableZOrder() const { return !((GetParam() & TestSuiteConfig::TextureDisableZOrder) == TestSuiteConfig::TextureDisableZOrder); }
		bool Force32BitIndices() const { return (GetParam() & TestSuiteConfig::Force32BitIndices) == TestSuiteConfig::Force32BitIndices; }
		bool TextureAsUNORM8() const { return (GetParam() & TestSuiteConfig::TextureAsUNORM8) == TestSuiteConfig::TextureAsUNORM8; }
		bool EnableAlphaCutoff() const { return (GetParam() & TestSuiteConfig::AlphaCutoff) == TestSuiteConfig::AlphaCutoff; }
		bool TestSerialization() const { return (GetParam() & TestSuiteConfig::Serialize) == TestSuiteConfig::Serialize; }
		
		omm::Cpu::Texture CreateTexture(const omm::Cpu::TextureDesc& desc) {
			omm::Cpu::Texture tex = 0;
			EXPECT_EQ(omm::Cpu::CreateTexture(_baker, desc, &tex), omm::Result::SUCCESS);
			_textures.push_back(tex);
			return tex;
		}

		void ExpectEqual(const omm::Debug::Stats& stats, const omm::Debug::Stats& expectedStats) {
			EXPECT_EQ(stats.totalOpaque, expectedStats.totalOpaque);
			EXPECT_EQ(stats.totalTransparent, expectedStats.totalTransparent);
			EXPECT_EQ(stats.totalUnknownTransparent, expectedStats.totalUnknownTransparent);
			EXPECT_EQ(stats.totalUnknownOpaque, expectedStats.totalUnknownOpaque);
			EXPECT_EQ(stats.totalFullyOpaque, expectedStats.totalFullyOpaque);
			EXPECT_EQ(stats.totalFullyTransparent, expectedStats.totalFullyTransparent);
			EXPECT_EQ(stats.totalFullyUnknownOpaque, expectedStats.totalFullyUnknownOpaque);
			EXPECT_EQ(stats.totalFullyUnknownTransparent, expectedStats.totalFullyUnknownTransparent);
		}

		std::vector<uint32_t> ConvertTexCoords(omm::TexCoordFormat format, float* texCoords, uint32_t texCoordsSize)
		{
			if (format == omm::TexCoordFormat::UV16_FLOAT || format == omm::TexCoordFormat::UV16_UNORM)
			{
				uint32_t numTexCoordPairs = texCoordsSize / sizeof(float2);
				std::vector<uint32_t> texCoordData(numTexCoordPairs);
				for (uint i = 0; i < numTexCoordPairs; ++i)
				{
					glm::vec2 v;
					v.x = ((float*)texCoords)[2 * i + 0];
					v.y = ((float*)texCoords)[2 * i + 1];

					if (format == omm::TexCoordFormat::UV16_UNORM)
						texCoordData[i] = glm::packUnorm2x16(v);
					else if (format == omm::TexCoordFormat::UV16_FLOAT)
						texCoordData[i] = glm::packHalf2x16(v);
					else
						assert(false);
				}
				return texCoordData;
			}
			else {
				assert(false);
				return {};
			}
		}

		std::vector<std::byte> readFileData(const std::string& name) {
			std::ifstream inputFile(name, std::ios_base::binary);

			// Determine the length of the file by seeking
			// to the end of the file, reading the value of the
			// position indicator, and then seeking back to the beginning.
			inputFile.seekg(0, std::ios_base::end);
			auto length = inputFile.tellg();
			inputFile.seekg(0, std::ios_base::beg);

			// Make a buffer of the exact size of the file and read the data into it.
			std::vector<std::byte> buffer(length);
			inputFile.read(reinterpret_cast<char*>(buffer.data()), length);

			inputFile.close();
			return buffer;
		}

		struct BakeOutput
		{
			omm::Debug::Stats stats;
			std::vector<uint8_t> serializedInput;
			std::vector<uint8_t> serializedOutput;
		};

		BakeOutput Bake(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			uint32_t indexCount,
			uint32_t* triangleIndices,
			omm::TexCoordFormat texCoordFormat,
			void* texCoords,
			omm::Cpu::Texture tex,
			const Options opt = {})
		{
			BakeOutput output;

			omm::Cpu::BakeInputDesc desc;
			desc.texture = tex;
			desc.format = opt.format;
			desc.alphaMode = omm::AlphaMode::Test;
			desc.runtimeSamplerDesc.addressingMode = opt.addressingMode;
			desc.runtimeSamplerDesc.filter = omm::TextureFilterMode::Linear;
			desc.indexFormat = omm::IndexFormat::UINT_32;
			desc.indexBuffer = triangleIndices;
			desc.texCoords = texCoords;
			desc.texCoordFormat = texCoordFormat;
			desc.indexCount = indexCount;
			desc.maxSubdivisionLevel = subdivisionLevel;
			desc.alphaCutoff = alphaCutoff;
			desc.alphaCutoffLessEqual = opt.alphaCutoffLE;
			desc.alphaCutoffGreater = opt.alphaCutoffGT;
			desc.unknownStatePromotion = opt.unknownStatePromotion;
			desc.bakeFlags = (omm::Cpu::BakeFlags)((uint32_t)omm::Cpu::BakeFlags::EnableInternalThreads);
			desc.maxWorkloadSize = opt.maxWorkloadSize;
			desc.unresolvedTriState = opt.unresolvedTriState;
			if (opt.mergeSimilar)
				desc.bakeFlags = (omm::Cpu::BakeFlags)((uint32_t)desc.bakeFlags | (uint32_t)omm::Cpu::BakeFlags::EnableNearDuplicateDetection);
			if (Force32BitIndices())
				desc.bakeFlags = (omm::Cpu::BakeFlags)((uint32_t)desc.bakeFlags | (uint32_t)omm::Cpu::BakeFlags::Force32BitIndices);
			if (!opt.enableSpecialIndices)
				desc.bakeFlags = (omm::Cpu::BakeFlags)((uint32_t)desc.bakeFlags | (uint32_t)omm::Cpu::BakeFlags::DisableSpecialIndices);

			desc.dynamicSubdivisionScale = opt.dynamicSubdivisionScale;

			omm::Cpu::BakeResult res = nullptr;
			const omm::Cpu::BakeResultDesc* resDesc = nullptr;
			if (TestSerialization() || opt.forceCorruptedBlob || opt.forceSerializedOutput)
			{
				{
					omm::Cpu::DeserializedDesc dataToSerialize;
					dataToSerialize.numInputDescs = 1;
					dataToSerialize.inputDescs = &desc;
					dataToSerialize.flags = opt.serializeCompress ? omm::Cpu::SerializeFlags::Compress : omm::Cpu::SerializeFlags::None;

					// Serialize...
					omm::Cpu::SerializedResult serializedRes = 0;
					EXPECT_EQ(omm::Cpu::Serialize(_baker, dataToSerialize, &serializedRes), omm::Result::SUCCESS);
					EXPECT_NE(serializedRes, nullptr);

					// Get data blob
					const omm::Cpu::BlobDesc* blob = nullptr;
					EXPECT_EQ(omm::Cpu::GetSerializedResultDesc(serializedRes, &blob), omm::Result::SUCCESS);

					// Copy content.
					output.serializedInput = std::vector<uint8_t>((uint8_t*)blob->data, (uint8_t*)blob->data + blob->size);

					// Destroy
					EXPECT_EQ(omm::Cpu::DestroySerializedResult(serializedRes), omm::Result::SUCCESS);
				}
				
				// Now do deserialize again and call bake
				{
					omm::Cpu::BlobDesc blob;
					blob.data = output.serializedInput.data();
					blob.size = output.serializedInput.size();

					if (opt.forceCorruptedBlob)
					{
						blob.size -= sizeof(uint32_t);
					}

					omm::Cpu::DeserializedResult dRes = nullptr;
					if (opt.forceCorruptedBlob)
					{
						EXPECT_EQ(omm::Cpu::Deserialize(_baker, blob, &dRes), omm::Result::INVALID_ARGUMENT);
						return BakeOutput();
					}
					else
					{
						EXPECT_EQ(omm::Cpu::Deserialize(_baker, blob, &dRes), omm::Result::SUCCESS);
					}

					EXPECT_NE(dRes, nullptr);

					const omm::Cpu::DeserializedDesc* desDesc = nullptr;
					EXPECT_EQ(omm::Cpu::GetDeserializedDesc(dRes, &desDesc), omm::Result::SUCCESS);
					
					EXPECT_EQ(desDesc->numInputDescs, 1);
					EXPECT_EQ(desDesc->numResultDescs, 0);

					const omm::Cpu::BakeInputDesc& descCopy = desDesc->inputDescs[0];

					EXPECT_EQ(omm::Cpu::Bake(_baker, descCopy, &res), opt.bakeResult);

					if (opt.bakeResult != omm::Result::SUCCESS)
					{
						return BakeOutput();
					}

					EXPECT_NE(res, nullptr);

					EXPECT_EQ(omm::Cpu::DestroyDeserializedResult(dRes), omm::Result::SUCCESS);

					EXPECT_EQ(omm::Cpu::GetBakeResultDesc(res, &resDesc), omm::Result::SUCCESS);
				}

				// Now try serialize the bake results too!
				{
					omm::Cpu::DeserializedDesc dataToSerialize;
					dataToSerialize.numResultDescs = 1;
					dataToSerialize.resultDescs = resDesc;
					dataToSerialize.flags = opt.serializeCompress ? omm::Cpu::SerializeFlags::Compress : omm::Cpu::SerializeFlags::None;

					// Serialize...
					omm::Cpu::SerializedResult serializedRes = 0;
					EXPECT_EQ(omm::Cpu::Serialize(_baker, dataToSerialize, &serializedRes), omm::Result::SUCCESS);
					EXPECT_NE(serializedRes, nullptr);

					// Get data blob
					const omm::Cpu::BlobDesc* blob = nullptr;
					EXPECT_EQ(omm::Cpu::GetSerializedResultDesc(serializedRes, &blob), omm::Result::SUCCESS);

					// Copy content.
					output.serializedOutput = std::vector<uint8_t>((uint8_t*)blob->data, (uint8_t*)blob->data + blob->size);

					// Destroy
					EXPECT_EQ(omm::Cpu::DestroySerializedResult(serializedRes), omm::Result::SUCCESS);
				}

				// Now do deserialize & compare
				{
					omm::Cpu::BlobDesc blob;
					blob.data = output.serializedOutput.data();
					blob.size = output.serializedOutput.size();

					omm::Cpu::DeserializedResult dRes = nullptr;
					EXPECT_EQ(omm::Cpu::Deserialize(_baker, blob, &dRes), omm::Result::SUCCESS);
					EXPECT_NE(dRes, nullptr);

					const omm::Cpu::DeserializedDesc* desDesc = nullptr;
					EXPECT_EQ(omm::Cpu::GetDeserializedDesc(dRes, &desDesc), omm::Result::SUCCESS);

					EXPECT_EQ(desDesc->numInputDescs, 0);
					EXPECT_EQ(desDesc->numResultDescs, 1);

					const omm::Cpu::BakeResultDesc* resDescCpy = &desDesc->resultDescs[0];
					// Compare results

					EXPECT_EQ(resDesc->arrayDataSize, resDescCpy->arrayDataSize);
					EXPECT_EQ(memcmp(resDesc->arrayData, resDescCpy->arrayData, resDesc->arrayDataSize), 0);

					EXPECT_EQ(resDesc->descArrayCount, resDescCpy->descArrayCount);
					EXPECT_EQ(memcmp(resDesc->descArray, resDescCpy->descArray, sizeof(omm::Cpu::OpacityMicromapDesc) * resDesc->descArrayCount), 0);

					EXPECT_EQ(resDesc->descArrayHistogramCount, resDescCpy->descArrayHistogramCount);
					EXPECT_EQ(memcmp(resDesc->descArrayHistogram, resDescCpy->descArrayHistogram, sizeof(omm::Cpu::OpacityMicromapUsageCount) * resDesc->descArrayHistogramCount), 0);

					EXPECT_EQ(resDesc->indexCount, resDescCpy->indexCount);
					EXPECT_EQ(resDesc->indexFormat, resDescCpy->indexFormat);
					size_t indexBufferFormatSize;
					if (resDescCpy->indexFormat == omm::IndexFormat::UINT_8)
						indexBufferFormatSize = 1;
					else if (resDescCpy->indexFormat == omm::IndexFormat::UINT_16)
						indexBufferFormatSize = 2;
					else // omm::IndexFormat::UINT_32
						indexBufferFormatSize = 4;
					EXPECT_EQ(memcmp(resDesc->indexBuffer, resDescCpy->indexBuffer, indexBufferFormatSize * resDesc->indexCount), 0);

					EXPECT_EQ(resDesc->indexHistogramCount, resDescCpy->indexHistogramCount);
					EXPECT_EQ(memcmp(resDesc->indexHistogram, resDescCpy->indexHistogram, sizeof(omm::Cpu::OpacityMicromapUsageCount)* resDesc->indexHistogramCount), 0);

					EXPECT_EQ(omm::Cpu::DestroyDeserializedResult(dRes), omm::Result::SUCCESS);
				}
			}
			else
			{
				EXPECT_EQ(omm::Cpu::Bake(_baker, desc, &res), opt.bakeResult);
				if (opt.bakeResult != omm::Result::SUCCESS)
				{
					return BakeOutput();
				}

				EXPECT_NE(res, nullptr);

				EXPECT_EQ(omm::Cpu::GetBakeResultDesc(res, &resDesc), omm::Result::SUCCESS);
			}

#if OMM_TEST_ENABLE_IMAGE_DUMP
			constexpr bool kDumpDebug = true;
#else
			constexpr bool kDumpDebug = false;
#endif

			if (kDumpDebug) 
			{
				std::string name = ::testing::UnitTest::GetInstance()->current_test_suite()->name();
				std::string tname = ::testing::UnitTest::GetInstance()->current_test_info()->name();
				std::string fileName = name + "_" + tname;
				std::replace(fileName.begin(), fileName.end(), '/', '_');

				omm::Debug::SaveAsImages(_baker, desc, resDesc, 
					{	.path = "OmmBakeOutput", 
						.filePostfix = fileName.c_str(), 
						.detailedCutout = opt.detailedCutout,
						.dumpOnlyFirstOMM = false,
						.monochromeUnknowns = opt.monochromeUnknowns,
						.oneFile = opt.oneFile });
			}

			// Manually collected stats from parsing the 
			if (resDesc)
			{
				EXPECT_EQ(omm::Debug::GetStats(_baker, resDesc, &output.stats), omm::Result::SUCCESS);
			}

			omm::Test::ValidateHistograms(resDesc);

			EXPECT_EQ(omm::Cpu::DestroyBakeResult(res), omm::Result::SUCCESS);

			return output;
		}

		omm::Debug::Stats Bake(const std::vector<uint8_t>& serializedInput, const Options opt = {})
		{
			omm::Cpu::BlobDesc blob;
			blob.data = (void*)serializedInput.data();
			blob.size = serializedInput.size();

			if (opt.forceCorruptedBlob)
			{
				blob.size -= sizeof(uint32_t);
			}

			omm::Cpu::DeserializedResult dRes = nullptr;
			if (opt.forceCorruptedBlob)
			{
				EXPECT_EQ(omm::Cpu::Deserialize(_baker, blob, &dRes), omm::Result::INVALID_ARGUMENT);
				return omm::Debug::Stats();
			}
			else
			{
				EXPECT_EQ(omm::Cpu::Deserialize(_baker, blob, &dRes), omm::Result::SUCCESS);
			}

			EXPECT_NE(dRes, nullptr);

			const omm::Cpu::DeserializedDesc* desDesc = nullptr;
			EXPECT_EQ(omm::Cpu::GetDeserializedDesc(dRes, &desDesc), omm::Result::SUCCESS);

			EXPECT_EQ(desDesc->numInputDescs, 1);
			EXPECT_EQ(desDesc->numResultDescs, 0);

			const omm::Cpu::BakeInputDesc& desc = desDesc->inputDescs[0];

			omm::Cpu::BakeResult res = nullptr;
			EXPECT_EQ(omm::Cpu::Bake(_baker, desc, &res), opt.bakeResult);

			if (opt.bakeResult != omm::Result::SUCCESS)
			{
				return omm::Debug::Stats();
			}

			EXPECT_NE(res, nullptr);

			EXPECT_EQ(omm::Cpu::DestroyDeserializedResult(dRes), omm::Result::SUCCESS);

			const omm::Cpu::BakeResultDesc* resDesc = nullptr;
			EXPECT_EQ(omm::Cpu::GetBakeResultDesc(res, &resDesc), omm::Result::SUCCESS); 

#if OMM_TEST_ENABLE_IMAGE_DUMP
			constexpr bool kDumpDebug = true;
#else
			constexpr bool kDumpDebug = false;
#endif

			if (kDumpDebug)
			{
				std::string name = ::testing::UnitTest::GetInstance()->current_test_suite()->name();
				std::string tname = ::testing::UnitTest::GetInstance()->current_test_info()->name();
				std::string fileName = name + "_" + tname;
				std::replace(fileName.begin(), fileName.end(), '/', '_');

				omm::Debug::SaveAsImages(_baker, desc, resDesc,
					{ .path = "OmmBakeOutput",
						.filePostfix = fileName.c_str(),
						.detailedCutout = opt.detailedCutout,
						.dumpOnlyFirstOMM = false,
						.monochromeUnknowns = opt.monochromeUnknowns,
						.oneFile = opt.oneFile });
			}

			// Manually collected stats from parsing the
			omm::Debug::Stats stats;
			if (resDesc)
			{
				EXPECT_EQ(omm::Debug::GetStats(_baker, resDesc, &stats), omm::Result::SUCCESS);
			}

			omm::Test::ValidateHistograms(resDesc);

			EXPECT_EQ(omm::Cpu::DestroyBakeResult(res), omm::Result::SUCCESS);

			return stats;
		}

		omm::Debug::Stats GetBakeOutput(const std::vector<uint8_t>& serializedOutput, const Options opt = {})
		{
			omm::Cpu::BlobDesc blob;
			blob.data = (void*)serializedOutput.data();
			blob.size = serializedOutput.size();

			if (opt.forceCorruptedBlob)
			{
				blob.size -= sizeof(uint32_t);
			}

			omm::Cpu::DeserializedResult dRes = nullptr;
			if (opt.forceCorruptedBlob)
			{
				EXPECT_EQ(omm::Cpu::Deserialize(_baker, blob, &dRes), omm::Result::INVALID_ARGUMENT);
				return omm::Debug::Stats();
			}
			else
			{
				EXPECT_EQ(omm::Cpu::Deserialize(_baker, blob, &dRes), omm::Result::SUCCESS);
			}

			EXPECT_NE(dRes, nullptr);

			const omm::Cpu::DeserializedDesc* desDesc = nullptr;
			EXPECT_EQ(omm::Cpu::GetDeserializedDesc(dRes, &desDesc), omm::Result::SUCCESS);

			EXPECT_EQ(desDesc->numInputDescs, 0);
			EXPECT_EQ(desDesc->numResultDescs, 1);

			const omm::Cpu::BakeResultDesc* resDesc = &desDesc->resultDescs[0];

			// Manually collected stats from parsing the
			omm::Debug::Stats stats;
			if (resDesc)
			{
				EXPECT_EQ(omm::Debug::GetStats(_baker, resDesc, &stats), omm::Result::SUCCESS);
			}

			omm::Test::ValidateHistograms(resDesc);

			EXPECT_EQ(omm::Cpu::DestroyDeserializedResult(dRes), omm::Result::SUCCESS);

			return stats;
		}

		omm::Debug::Stats GetOmmBakeStats(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			uint32_t indexCount,
			uint32_t* triangleIndices,
			omm::TexCoordFormat texCoordFormat,
			void* texCoords,
			omm::Cpu::Texture tex,
			const Options opt = {})
		{
			BakeOutput res = Bake(alphaCutoff, subdivisionLevel, texSize, indexCount, triangleIndices, texCoordFormat, texCoords, tex, opt);
			return res.stats;
		}

		BakeOutput GetOmmBakeOutput(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			uint32_t indexCount,
			uint32_t* triangleIndices,
			omm::TexCoordFormat texCoordFormat,
			void* texCoords,
			omm::Cpu::Texture tex,
			const Options opt = {})
		{
			return Bake(alphaCutoff, subdivisionLevel, texSize, indexCount, triangleIndices, texCoordFormat, texCoords, tex, opt);
		}

		omm::Debug::Stats GetOmmBakeStatsFP32(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			uint32_t indexCount,
			uint32_t* triangleIndices,
			omm::TexCoordFormat texCoordFormat,
			void* texCoords,
			std::function<float(int i, int j, int w, int h, int mip)> tex,
			const Options opt = {})
		{
			vmtest::TextureFP32 texture(texSize.x, texSize.y, opt.mipCount, EnableZOrder(), EnableAlphaCutoff() ? alphaCutoff : -1.f, tex);
			omm::Cpu::Texture texHandle = CreateTexture(texture.GetDesc());
			return GetOmmBakeStats(alphaCutoff, subdivisionLevel, texSize, indexCount, triangleIndices, texCoordFormat, texCoords, texHandle, opt);
		}

		BakeOutput GetOmmBakeOutputFP32(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			uint32_t indexCount,
			uint32_t* triangleIndices,
			omm::TexCoordFormat texCoordFormat,
			void* texCoords,
			std::function<float(int i, int j, int w, int h, int mip)> tex,
			const Options opt = {})
		{
			vmtest::TextureFP32 texture(texSize.x, texSize.y, opt.mipCount, EnableZOrder(), EnableAlphaCutoff() ? alphaCutoff : -1.f, tex);
			omm::Cpu::Texture texHandle = CreateTexture(texture.GetDesc());
			return GetOmmBakeOutput(alphaCutoff, subdivisionLevel, texSize, indexCount, triangleIndices, texCoordFormat, texCoords, texHandle, opt);
		}

		omm::Debug::Stats GetOmmBakeStatsFP32(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			std::function<float(int i, int j, int w, int h, int mip)> tex,
			const Options opt = {})
		{
			uint32_t triangleIndices[6] = { 0, 1, 2, 3, 1, 2 };
			float texCoords[8] = { 0.f, 0.f,	0.f, 1.f,	1.f, 0.f,	 1.f, 1.f };
			return GetOmmBakeStatsFP32(alphaCutoff, subdivisionLevel, texSize, 6, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, tex, opt);
		}

		BakeOutput GetOmmBakeOutputFP32(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			std::function<float(int i, int j, int w, int h, int mip)> tex,
			const Options opt = {})
		{
			uint32_t triangleIndices[6] = { 0, 1, 2, 3, 1, 2 };
			float texCoords[8] = { 0.f, 0.f,	0.f, 1.f,	1.f, 0.f,	 1.f, 1.f };
			return GetOmmBakeOutputFP32(alphaCutoff, subdivisionLevel, texSize, 6, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, tex, opt);
		}

		omm::Debug::Stats GetOmmBakeStatsUNORM8(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			uint32_t indexCount,
			uint32_t* triangleIndices,
			float* texCoords,
			std::function<uint8_t(int i, int j, int w, int h, int mip)> tex,
			const Options opt = {})
		{
			vmtest::TextureUNORM8 texture(texSize.x, texSize.y, opt.mipCount, EnableZOrder(), EnableAlphaCutoff() ? alphaCutoff : -1.f, tex);
			omm::Cpu::Texture texHandle = CreateTexture(texture.GetDesc());
			return GetOmmBakeStats(alphaCutoff, subdivisionLevel, texSize, indexCount, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, texHandle, opt);
		}

		omm::Debug::Stats GetOmmBakeStatsUNORM8(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			std::function<uint8_t(int i, int j, int w, int h, int mip)> tex,
			const Options opt = {})
		{
			uint32_t triangleIndices[6] = { 0, 1, 2, 3, 1, 2 };
			float texCoords[8] = { 0.f, 0.f,	0.f, 1.f,	1.f, 0.f,	 1.f, 1.f };
			vmtest::TextureUNORM8 texture(texSize.x, texSize.y, opt.mipCount, EnableZOrder(), EnableAlphaCutoff() ? alphaCutoff : -1.f, tex);
			omm::Cpu::Texture texHandle = CreateTexture(texture.GetDesc());
			return GetOmmBakeStats(alphaCutoff, subdivisionLevel, texSize, 6, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, texHandle, opt);
		}

		omm::Debug::Stats LeafletMipN(uint32_t mipStart, uint32_t NumMip, float alphaCutoff = 0.5f)
		{
			uint32_t subdivisionLevel = 6;
			uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

			uint32_t indexCount = 3;
			uint32_t triangleIndices[3] = { 0, 1, 2 };
			//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
			float texCoords[6] = { 0.05f, 0.1f,  0.1f, 0.9f,  0.9f, 0.9f };

			int width, height, channels;
			unsigned char* pixelData = stbi_load(PROJECT_SOURCE_DIR "/assets/tests/leaflet.png", &width, &height, &channels, 0);

			const uint32_t NumMipToGenerate = mipStart + NumMip;
			std::vector<std::vector<float>> mips;
			std::vector<std::tuple<uint32_t, uint32_t>> mipDims;
			mips.resize(NumMipToGenerate);
			mipDims.resize(NumMipToGenerate);

			mips[0].reserve(width * height);
			mipDims[0] = std::make_tuple(width, height);

			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < width; i++)
				{
					uint8_t* pixel = pixelData + j * width * channels + channels * i + 2;
					mips[0].push_back(*pixel / 255.f);
				}
			}

			auto GenerateMipAvgFilter = [](std::vector<float> tex, int32_t& w, int32_t& h)->std::vector<float>
			{
				int32_t halfW = w / 2;
				int32_t halfH = h / 2;
				std::vector<float> mip;
				mip.reserve(halfW * halfH);

				for (int32_t j = 0; j < halfH; ++j)
				{
					for (int32_t i = 0; i < halfW; ++i)
					{
						float pixel0 = tex[2 * j * w + 2 * i];
						float pixel1 = tex[(2 * j + 1) * w + 2 * i];
						float pixel2 = tex[2 * j * w + (2 * i + 1)];
						float pixel3 = tex[(2 * j + 1) * w + (2 * i + 1)];

						const float pixel = (pixel0 + pixel1 + pixel2 + pixel3) * 0.25f;
						mip.push_back(pixel);
					}
				}

				w = halfW;
				h = halfH;
				return mip;
			};

			int mipWidth = width;
			int mipHeight = height;
			for (uint32_t i = 1; i < NumMipToGenerate; ++i)
			{
				mips[i] = GenerateMipAvgFilter(mips[i - 1], mipWidth, mipHeight);
				mipDims[i] = std::make_tuple(mipWidth, mipHeight);
			}

			int2 size = { std::get<0>(mipDims[mipStart]), std::get<1>(mipDims[mipStart]) };
			omm::Debug::Stats stats = GetOmmBakeStatsFP32(alphaCutoff, subdivisionLevel, size, indexCount, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, [mipStart, mips](int i, int j, int w, int h, int mip)->float {

				return 1.f - mips[mipStart + mip][w * j + i];

				}, { .format = omm::Format::OC1_4_State, .mipCount = NumMip, .oneFile = false, .detailedCutout = false });

			return stats;
		}

		omm::Debug::Stats LeafletLevelN(uint32_t subdivisionLevel, uint64_t maxWorkloadSize = 0xFFFFFFFFFFFFFFFF, omm::Result bakeResult = omm::Result::SUCCESS)
		{
			uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

			uint32_t indexCount = 3;
			uint32_t triangleIndices[3] = { 0, 1, 2 };
			//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
			float texCoords[6] = { 0.35f, 0.1f,  0.1f, 0.9f,  0.9f, 0.8f };

			int width, height, channels;
			unsigned char* pixelData = stbi_load(PROJECT_SOURCE_DIR "/assets/tests/leaflet.png", &width, &height, &channels, 0);

			std::vector<float> mips;

			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < width; i++)
				{
					uint8_t* pixel = pixelData + j * width * channels + channels * i + 2;
					mips.push_back(*pixel / 255.f);
				}
			}

			int2 size = { width, height };
			omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, size, indexCount, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, [mips](int i, int j, int w, int h, int mip)->float {

				return 1.f - mips[w * j + i];

				}, { .format = omm::Format::OC1_4_State, .unknownStatePromotion = omm::UnknownStatePromotion::Nearest, .enableSpecialIndices = false, .oneFile = true, .detailedCutout = false, .maxWorkloadSize = maxWorkloadSize, .bakeResult = bakeResult });

			return stats;
		}

		void GenerateSerializedString(const char* inputStr, const char* outputStr, bool compress)
		{
			auto binaryToHex = [](const char* name, const std::vector<unsigned char>& binaryData) {
				std::cout << name << " = {";

				for (size_t i = 0; i < binaryData.size(); ++i) {
					std::cout << "0x" << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << (int)binaryData[i];

					if (i != binaryData.size() - 1) {
						std::cout << ", ";
					}

					if ((i + 1) % 20 == 0 && i != binaryData.size() - 1) {
						std::cout << std::endl;
					}
				}

				std::cout << "};" << std::endl;
				};

			uint32_t subdivisionLevel = 4;
			uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

			BakeOutput output = GetOmmBakeOutputFP32(0.5f, subdivisionLevel, { 8, 8 }, &StandardCircle, { .forceSerializedOutput = true, .serializeCompress = compress });

			if (inputStr)
				binaryToHex(inputStr, output.serializedInput);
			if (outputStr)
				binaryToHex(outputStr, output.serializedOutput);
		}

		std::vector< omm::Cpu::Texture> _textures;
		omm::Baker _baker = 0;
	};

	TEST_P(OMMBakeTestCPU, NullDesc) {

		omm::Cpu::BakeInputDesc nullDesc;
		omm::Cpu::BakeResult res = nullptr;
		EXPECT_EQ(omm::Cpu::Bake(_baker, nullDesc, &res), omm::Result::INVALID_ARGUMENT);
		EXPECT_EQ(res, nullptr);
	}

	TEST_P(OMMBakeTestCPU, AllOpaque4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, {1024, 1024}, [](int i, int j, int w, int h, int mip)->float {
			return 0.6f;
			});

		ExpectEqual(stats, { .totalFullyOpaque = 2 });
	}


	TEST_P(OMMBakeTestCPU, AllOpaque3) {

		uint32_t subdivisionLevel = 3;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.6f;
			});

		ExpectEqual(stats, { .totalFullyOpaque = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllOpaque2) {

		uint32_t subdivisionLevel = 2;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.6f;
			});

		ExpectEqual(stats, { .totalFullyOpaque = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllOpaque1) {

		uint32_t subdivisionLevel = 1;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.6f;
			});

		ExpectEqual(stats, { .totalFullyOpaque = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllOpaque0) {

		uint32_t subdivisionLevel = 0;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.6f;
			});

		ExpectEqual(stats, { .totalFullyOpaque = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllTransparent4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.4f;
			});

		ExpectEqual(stats, { .totalFullyTransparent = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllTransparent3) {

		uint32_t subdivisionLevel = 3;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.4f;
			});

		ExpectEqual(stats, { .totalFullyTransparent = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllTransparent2) {

		uint32_t subdivisionLevel = 2;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.4f;
			});

		ExpectEqual(stats, { .totalFullyTransparent = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllTransparent1) {

		uint32_t subdivisionLevel = 1;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.4f;
			});

		ExpectEqual(stats, { .totalFullyTransparent = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllUnknownTransparent) {

		uint32_t subdivisionLevel = 1;

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if ((i) % 8 != (j) % 8)
				return 0.f;
			else
				return 1.f;
			});

		ExpectEqual(stats, { .totalFullyUnknownTransparent = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllUnknownOpaque) {

		uint32_t subdivisionLevel = 1;

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if ((i) % 8 != (j) % 8)
				return 1.f;
			else
				return 0.f;
			});

		ExpectEqual(stats, { .totalFullyUnknownOpaque = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllTransparentOpaqueCorner4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if (i == 0 && j == 0)
				return 0.6f;
			return 0.4f;
			});

		ExpectEqual(stats, {
			.totalTransparent = numMicroTris - 1,
			.totalUnknownTransparent = 1,
			.totalFullyTransparent = 1,
			});
	}

	TEST_P(OMMBakeTestCPU, AllOpaque1_CorruptedBlob) {

		uint32_t subdivisionLevel = 1;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.6f;
			}, { .bakeResult = omm::Result::INVALID_ARGUMENT, .forceCorruptedBlob = true });

		ExpectEqual(stats, { .totalFullyOpaque = 0 });
	}

	TEST_P(OMMBakeTestCPU, Circle) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, &StandardCircle);

		ExpectEqual(stats, {
			.totalOpaque = 204,
			.totalTransparent = 219,
			.totalUnknownTransparent = 39,
			.totalUnknownOpaque = 50,
			});
	}

	TEST_P(OMMBakeTestCPU, CircleMergeSimilar) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, &StandardCircle, { .mergeSimilar = true });

		ExpectEqual(stats, {
			.totalOpaque = 200,
			.totalTransparent = 216,
			.totalUnknownTransparent = 42,
			.totalUnknownOpaque = 54,
			});
	}

	TEST_P(OMMBakeTestCPU, CircleOC2) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, &StandardCircle, { .format = omm::Format::OC1_2_State });

		ExpectEqual(stats, {
			.totalOpaque = 254,
			.totalTransparent = 258,
			});
	}

	TEST_P(OMMBakeTestCPU, SineUNORM8) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsUNORM8(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->uint8_t {
			const float uv = float(i) / (float)w;
			const float val = 0.5f - 0.5f * std::sin(uv * 15);
			const uint8_t val8 = (uint8_t)(val * 255.f);
			return val8;
			});

		ExpectEqual(stats, {
			.totalOpaque = 128,
			.totalTransparent = 256,
			.totalUnknownTransparent = 48,
			.totalUnknownOpaque = 80,
			});
	}

	TEST_P(OMMBakeTestCPU, Sine) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)w;

			return 1.f - std::sin(uv*15);
			});

		ExpectEqual(stats, {
			.totalOpaque = 224,
			.totalTransparent = 128,
			.totalUnknownTransparent = 96,
			.totalUnknownOpaque = 64,
			});
	}

	TEST_P(OMMBakeTestCPU, SineOC2) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)w;

			return 1.f - std::sin(uv * 15);
			}, { .format = omm::Format::OC1_2_State });

		ExpectEqual(stats, {
			.totalOpaque = 288,
			.totalTransparent = 224,
			});
	}

	TEST_P(OMMBakeTestCPU, SineOC2Neg) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)w;

			return 1.f - std::sin(uv * 15);
			}, { .format = omm::Format::OC1_2_State });

		ExpectEqual(stats, {
			.totalOpaque = 288,
			.totalTransparent = 224,
			});
	}

	TEST_P(OMMBakeTestCPU, Mandelbrot) {

		uint32_t subdivisionLevel = 5;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			
			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(w, h) - 0.1f;
			float2 coord = 2.f * uv - 1.f;
			float2 z = float2(0, 0);
			float2 c = coord - float2(0.5, 0);
			bool inMandelbrotSet = true;

			for (int i = 0; i < 20; i++) {
				z = complexMultiply(z, z) + c;
				if (length(z) > 2.) {
					inMandelbrotSet = false;
					break;
				}
			}
			if (inMandelbrotSet) {
				return 0.f;
			}
			else { 
				return 1.f;
			}

		}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 1212,
			.totalTransparent = 484,
			.totalUnknownTransparent = 124,
			.totalUnknownOpaque = 228,
			});
	}

	TEST_P(OMMBakeTestCPU, Mandelbrot2) {

		uint32_t subdivisionLevel = 5;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 3;
		uint32_t triangleIndices[6] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, [](int i, int j, int w, int h, int mip)->float {

			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(w, h) - 0.1f;
			float2 coord = 2.f * uv - 1.f;
			float2 z = float2(0, 0);
			float2 c = coord - float2(0.5, 0);
			bool inMandelbrotSet = true;

			for (int i = 0; i < 20; i++) {
				z = complexMultiply(z, z) + c;
				if (length(z) > 2.) {
					inMandelbrotSet = false;
					break;
				}
			}
			if (inMandelbrotSet) {
				return 0.f;
			}
			else {
				return 1.f;
			}

			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 521,
			.totalTransparent = 286,
			.totalUnknownTransparent = 82,
			.totalUnknownOpaque = 135,
			});
	}

	TEST_P(OMMBakeTestCPU, Mandelbrot3) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 3;
		uint32_t triangleIndices[6] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, [](int i, int j, int w, int h, int mip)->float {

			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(w, h) - 0.1f;
			float2 coord = 2.f * uv - 1.f;
			float2 z = float2(0, 0);
			float2 c = coord - float2(0.5, 0);
			bool inMandelbrotSet = true;

			for (int i = 0; i < 20; i++) {
				z = complexMultiply(z, z) + c;
				if (length(z) > 2.) {
					inMandelbrotSet = false;
					break;
				}
			}
			if (inMandelbrotSet) {
				return 0.f;
			}
			else {
				return 1.f;
			}

			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 164040,
			.totalTransparent = 91320,
			.totalUnknownTransparent = 3039,
			.totalUnknownOpaque = 3745,
			});
	}

	float GetJulia(int i, int j, int w, int h, int mip)
	{
		auto multiply = [](float2 x, float2 y)->float2 {
			return float2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
		};

		float2 uv = 1.2f * float2(i, j) / float2(w, h) - 0.1f;

		float2 z0 = 5.f * (uv - float2(.5f, .27f));
		float2 col;
		float time = 3.1f;
		float2 c = std::cos(time) * float2(std::cos(time / 2.f), std::sin(time / 2.f));
		for (int i = 0; i < 500; i++) {
			float2 z = multiply(z0, z0) + c;
			float mq = dot(z, z);
			if (mq > 4.f) {
				col = float2(float(i) / 20.f, 0.f);
				break;
			}
			else {
				z0 = z;
			}
			col = float2(mq / 2.f, mq / 2.f);
		}

		float alpha = std::clamp(col.x, 0.f, 1.f) >= 0.5f ? 0.6f : 0.4f;
		return 1.f - alpha;
	}

	TEST_P(OMMBakeTestCPU, Julia) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 3;
		uint32_t triangleIndices[6] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, [](int i, int j, int w, int h, int mip)->float {

			return GetJulia(i, j, w, h, mip);

			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 254265, 
			.totalTransparent = 5055,
			.totalUnknownTransparent = 1336,
			.totalUnknownOpaque = 1488,
				});
	}

	TEST_P(OMMBakeTestCPU, Julia_UVFP16) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 3;
		uint32_t triangleIndices[6] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };
		std::vector<uint32_t> texCoordsFP16 = ConvertTexCoords(omm::TexCoordFormat::UV16_FLOAT, texCoords, sizeof(texCoords));

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, omm::TexCoordFormat::UV16_FLOAT, texCoordsFP16.data(), [](int i, int j, int w, int h, int mip)->float {

			return GetJulia(i, j, w, h, mip);

			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
				.totalOpaque = 254321,
				.totalTransparent = 5108,
				.totalUnknownTransparent = 1264,
				.totalUnknownOpaque = 1451,
			});
	}

	TEST_P(OMMBakeTestCPU, Julia_UV_UNORM16) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 3;
		uint32_t triangleIndices[6] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };
		std::vector<uint32_t> texCoordsUnorm16 = ConvertTexCoords(omm::TexCoordFormat::UV16_UNORM, texCoords, sizeof(texCoords));

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, omm::TexCoordFormat::UV16_UNORM, texCoordsUnorm16.data(), [](int i, int j, int w, int h, int mip)->float {

			return GetJulia(i, j, w, h, mip);

			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
				.totalOpaque = 254325,
				.totalTransparent = 5110,
				.totalUnknownTransparent = 1284,
				.totalUnknownOpaque = 1425,
			});
	}

	TEST_P(OMMBakeTestCPU, JuliaUNORM8) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 3;
		uint32_t triangleIndices[6] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = GetOmmBakeStatsUNORM8(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, texCoords, [](int i, int j, int w, int h, int mip)->uint8_t {
			const float val = GetJulia(i, j, w, h, mip);
			return (uint8_t)std::clamp(val * 255.f, 0.f, 255.f) ;

			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 254251,
			.totalTransparent = 5176,
			.totalUnknownTransparent = 1215,
			.totalUnknownOpaque = 1502,
			});
	}

	TEST_P(OMMBakeTestCPU, Julia_T_AND_UO) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 3;
		uint32_t triangleIndices[6] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		Options ops;
		ops.alphaCutoffLE = omm::OpacityState::Transparent;
		ops.alphaCutoffGT = omm::OpacityState::UnknownOpaque;

		omm::Debug::Stats stats = GetOmmBakeStatsUNORM8(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, texCoords, [](int i, int j, int w, int h, int mip)->uint8_t {
			const float val = GetJulia(i, j, w, h, mip);
			return (uint8_t)std::clamp(val * 255.f, 0.f, 255.f);
			}, ops);

		ExpectEqual(stats, {
		    .totalOpaque = 0,
		    .totalTransparent = 5176,
		    .totalUnknownTransparent = 1215,
		    .totalUnknownOpaque = 1502 + 254251,
		});
	}

	TEST_P(OMMBakeTestCPU, Julia_FLIP_T_AND_O) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 3;
		uint32_t triangleIndices[6] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		Options ops;
		ops.alphaCutoffLE = omm::OpacityState::Opaque;
		ops.alphaCutoffGT = omm::OpacityState::Transparent;

		omm::Debug::Stats stats = GetOmmBakeStatsUNORM8(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, texCoords, [](int i, int j, int w, int h, int mip)->uint8_t {
			const float val = GetJulia(i, j, w, h, mip);
			return (uint8_t)std::clamp(val * 255.f, 0.f, 255.f);
			}, ops);

		ExpectEqual(stats, {
					.totalOpaque = 5176,
					.totalTransparent = 254251,
					.totalUnknownTransparent = 1502,
					.totalUnknownOpaque = 1215,
			});
	}

	TEST_P(OMMBakeTestCPU, Uniform) {

		uint32_t subdivisionLevel = 6;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 6;
		uint32_t triangleIndices[6] = { 0, 1, 2, 1, 2, 3, };
		//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
		float texCoords[8] = { 0.f, 0.f,  0.f, 1.0f,  1.f, 1.f, 1.f, 0.f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 4, 4 }, indexCount, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, [](int i, int j, int w, int h, int mip)->float {

			uint32_t x = (i) % 2;
			uint32_t y = (j) % 2;

			float values[4] = 
			{
				0.9f, 0.1f,
				0.1f, 0.7f
			};

			return 1.f - values[x + 2 *y];

			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 5132,
			.totalTransparent = 2393,
			.totalUnknownTransparent = 357,
			.totalUnknownOpaque = 310,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsLvl6) {

		uint32_t subdivisionLevel = 6;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 6;
		uint32_t triangleIndices[6] = { 0, 1, 2, 1, 2, 3, };
		//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
		float texCoords[8] = { 0.f, 0.f,  0.f, 1.0f,  1.f, 1.f, 1.f, 0.f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 902,
			.totalTransparent = 0,
			.totalUnknownTransparent = 3,
			.totalUnknownOpaque = 7287,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsLvl8) {

		uint32_t subdivisionLevel = 8;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 6;
		uint32_t triangleIndices[6] = { 0, 1, 2, 1, 2, 3, };
		//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
		float texCoords[8] = { 0.f, 0.f,  0.f, 1.0f,  1.f, 1.f, 1.f, 0.f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 77995,
			.totalTransparent = 535,
			.totalUnknownTransparent = 23163,
			.totalUnknownOpaque = 29379,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsReuseLvl2) {

		uint32_t subdivisionLevel = 2;

		std::vector<uint32_t> indices;
		std::vector<float2> texCoords;

		const uint32_t N = 32;
		const uint32_t M = 32;
		for (uint32_t j = 0; j < M; ++j)
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				const uint32_t indexOffset = 3 * (i + j * N);
				indices.push_back(indexOffset + 0);
				indices.push_back(indexOffset + 1);
				indices.push_back(indexOffset + 2);

				const float2 offset = float2(float(i) / float(N), float(j) / float(M));
				texCoords.push_back(offset + float2(0.f, 0.f) / float2(N, M));
				texCoords.push_back(offset + float2(0.f, 1.f) / float2(N, M));
				texCoords.push_back(offset + float2(1.f, 1.f) / float2(N, M));
			}
		}

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, (uint32_t)indices.size(), indices.data(), omm::TexCoordFormat::UV32_FLOAT, (float*)texCoords.data(), [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 6933,
			.totalUnknownTransparent = 1935,
			.totalUnknownOpaque = 7516,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsReuseLvl3) {

		uint32_t subdivisionLevel = 3;

		std::vector<uint32_t> indices;
		std::vector<float2> texCoords;

		const uint32_t N = 32;
		const uint32_t M = 32;
		for (uint32_t j = 0; j < M; ++j)
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				const uint32_t indexOffset = 3 * (i + j * N);
				indices.push_back(indexOffset + 0);
				indices.push_back(indexOffset + 1);
				indices.push_back(indexOffset + 2);

				const float2 offset = float2(float(i) / float(N), float(j) / float(M));
				texCoords.push_back(offset + float2(0.f, 0.f) / float2(N, M));
				texCoords.push_back(offset + float2(0.f, 1.f) / float2(N, M));
				texCoords.push_back(offset + float2(1.f, 1.f) / float2(N, M));
			}
		}

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, (uint32_t)indices.size(), indices.data(), omm::TexCoordFormat::UV32_FLOAT, (float*)texCoords.data(), [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 40134,
			.totalTransparent = 250,
			.totalUnknownTransparent = 11939,
			.totalUnknownOpaque = 13213,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsReuseLvl4) {

		uint32_t subdivisionLevel = 4;

		std::vector<uint32_t> indices;
		std::vector<float2> texCoords;

		const uint32_t N = 32;
		const uint32_t M = 32;
		for (uint32_t j = 0; j < M; ++j)
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				const uint32_t indexOffset = 3 * (i + j * N);
				indices.push_back(indexOffset + 0);
				indices.push_back(indexOffset + 1);
				indices.push_back(indexOffset + 2);

				const float2 offset = float2(float(i) / float(N), float(j) / float(M));
				texCoords.push_back(offset + float2(0.f, 0.f) / float2(N, M));
				texCoords.push_back(offset + float2(0.f, 1.f) / float2(N, M));
				texCoords.push_back(offset + float2(1.f, 1.f) / float2(N, M));
			}
		}

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, (uint32_t)indices.size(), indices.data(), omm::TexCoordFormat::UV32_FLOAT, (float*)texCoords.data(), [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 187129,
			.totalTransparent = 17979,
			.totalUnknownTransparent = 30309,
			.totalUnknownOpaque = 26727,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsReuseLvl5) {

		uint32_t subdivisionLevel = 5;

		std::vector<uint32_t> indices;
		std::vector<float2> texCoords;

		const uint32_t N = 32;
		const uint32_t M = 32;
		for (uint32_t j = 0; j < M; ++j)
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				const uint32_t indexOffset = 3 * (i + j * N);
				indices.push_back(indexOffset + 0);
				indices.push_back(indexOffset + 1);
				indices.push_back(indexOffset + 2);

				const float2 offset = float2(float(i) / float(N), float(j) / float(M));
				texCoords.push_back(offset + float2(0.f, 0.f) / float2(N, M));
				texCoords.push_back(offset + float2(0.f, 1.f) / float2(N, M));
				texCoords.push_back(offset + float2(1.f, 1.f) / float2(N, M));
			}
		}

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, (uint32_t)indices.size(), indices.data(), omm::TexCoordFormat::UV32_FLOAT, (float*)texCoords.data(), [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::Format::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 796515,
			.totalTransparent = 138195,
			.totalUnknownTransparent = 56743,
			.totalUnknownOpaque = 57123,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsReuseLSH) {

		uint32_t subdivisionLevel = 4;

		std::vector<uint32_t> indices;
		std::vector<float2> texCoords;

		const uint32_t N = 32;
		const uint32_t M = 32;
		for (uint32_t j = 0; j < M; ++j)
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				const uint32_t indexOffset = 3 * (i + j * N);
				indices.push_back(indexOffset + 0);
				indices.push_back(indexOffset + 1);
				indices.push_back(indexOffset + 2);

				const float2 offset = float2(float(i) / float(N), float(j) / float(M));
				texCoords.push_back(offset + float2(0.f, 0.f) / float2(N, M));
				texCoords.push_back(offset + float2(0.f, 1.f) / float2(N, M));
				texCoords.push_back(offset + float2(1.f, 1.f) / float2(N, M));
			}
		}

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, subdivisionLevel, { 1024, 1024 }, (uint32_t)indices.size(), indices.data(), omm::TexCoordFormat::UV32_FLOAT, (float*)texCoords.data(), [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::Format::OC1_4_State, .mergeSimilar = true });

		ExpectEqual(stats, {
			.totalOpaque = 170724,
			.totalTransparent = 11380,
			.totalUnknownTransparent = 37864,
			.totalUnknownOpaque = 39104,
			.totalFullyTransparent = 12,
			});
	}

	TEST_P(OMMBakeTestCPU, Leaflet_Alpha_0_2) {

		omm::Debug::Stats stats = LeafletMipN(0, 1, 0.2f);

		ExpectEqual(stats, {
			.totalOpaque = 864,
			.totalTransparent = 2712,
			.totalUnknownTransparent = 275,
			.totalUnknownOpaque = 245,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_0) {

		omm::Debug::Stats stats = LeafletMipN(0, 1);

		ExpectEqual(stats, {
			.totalOpaque = 817,
			.totalTransparent = 2763,
			.totalUnknownTransparent = 232,
			.totalUnknownOpaque = 284,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_1) {

		omm::Debug::Stats stats = LeafletMipN(0, 2);

		ExpectEqual(stats, {
			.totalOpaque = 809,
			.totalTransparent = 2720,
			.totalUnknownTransparent = 275,
			.totalUnknownOpaque = 292,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_2) {

		omm::Debug::Stats stats = LeafletMipN(0, 3);

		ExpectEqual(stats, {
			.totalOpaque = 784,
			.totalTransparent = 2688,
			.totalUnknownTransparent = 307,
			.totalUnknownOpaque = 317,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_3) {

		omm::Debug::Stats stats = LeafletMipN(0, 4);

		ExpectEqual(stats, {
			.totalOpaque = 776,
			.totalTransparent = 2684,
			.totalUnknownTransparent = 311,
			.totalUnknownOpaque = 325,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_4) {

		omm::Debug::Stats stats = LeafletMipN(0, 5);

		ExpectEqual(stats, {
			.totalOpaque = 724,
			.totalTransparent = 2586,
			.totalUnknownTransparent = 409,
			.totalUnknownOpaque = 377,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_5) {

		omm::Debug::Stats stats = LeafletMipN(0, 6);

		ExpectEqual(stats, {
			.totalOpaque = 615,
			.totalTransparent = 2430,
			.totalUnknownTransparent = 565 ,
			.totalUnknownOpaque = 486,
			});
	}


	TEST_P(OMMBakeTestCPU, LeafletMip0_to_6) {

		omm::Debug::Stats stats = LeafletMipN(0, 7);

		ExpectEqual(stats, {
			.totalOpaque = 349,
			.totalTransparent = 2408,
			.totalUnknownTransparent = 587,
			.totalUnknownOpaque = 752,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_7) {

		omm::Debug::Stats stats = LeafletMipN(0, 8);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 2408,
			.totalUnknownTransparent = 587,
			.totalUnknownOpaque = 1101,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0) {

		omm::Debug::Stats stats = LeafletMipN(0, 1);

		ExpectEqual(stats, {
			.totalOpaque = 817,
			.totalTransparent = 2763,
			.totalUnknownTransparent = 232,
			.totalUnknownOpaque = 284,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip1) {

		omm::Debug::Stats stats = LeafletMipN(1, 1);

		ExpectEqual(stats, {
			.totalOpaque = 847,
			.totalTransparent = 2728,
			.totalUnknownTransparent = 248,
			.totalUnknownOpaque = 273,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip2) {

		omm::Debug::Stats stats = LeafletMipN(2, 1);

		ExpectEqual(stats, {
			.totalOpaque = 857,
			.totalTransparent = 2725,
			.totalUnknownTransparent = 268,
			.totalUnknownOpaque = 246,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip3) {

		omm::Debug::Stats stats = LeafletMipN(3, 1);

		ExpectEqual(stats, {
			.totalOpaque = 867,
			.totalTransparent = 2735,
			.totalUnknownTransparent = 239,
			.totalUnknownOpaque = 255,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip4) {

		omm::Debug::Stats stats = LeafletMipN(4, 1);

		ExpectEqual(stats, {
			.totalOpaque = 928,
			.totalTransparent = 2777,
			.totalUnknownTransparent = 199,
			.totalUnknownOpaque = 192,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip5) {

		omm::Debug::Stats stats = LeafletMipN(5, 1);

		ExpectEqual(stats, {
			.totalOpaque = 965,
			.totalTransparent = 2821,
			.totalUnknownTransparent = 156,
			.totalUnknownOpaque = 154,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip6) {

		omm::Debug::Stats stats = LeafletMipN(6, 1);

		ExpectEqual(stats, {
			.totalOpaque = 526,
			.totalTransparent = 3335,
			.totalUnknownTransparent = 119,
			.totalUnknownOpaque = 116,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel0) {

		omm::Debug::Stats stats = LeafletLevelN(0);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 0,
			.totalUnknownTransparent = 1,
			.totalUnknownOpaque = 0,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel1) {

		omm::Debug::Stats stats = LeafletLevelN(1);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 0,
			.totalUnknownTransparent = 4,
			.totalUnknownOpaque = 0,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel2) {

		omm::Debug::Stats stats = LeafletLevelN(2);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 1,
			.totalUnknownTransparent = 10,
			.totalUnknownOpaque = 5,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel3) {

		omm::Debug::Stats stats = LeafletLevelN(3);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 16,
			.totalUnknownTransparent = 31,
			.totalUnknownOpaque = 17,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel4) {

		omm::Debug::Stats stats = LeafletLevelN(4);

		ExpectEqual(stats, {
			.totalOpaque = 35,
			.totalTransparent = 108,
			.totalUnknownTransparent = 68,
			.totalUnknownOpaque = 45,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel5) {

		omm::Debug::Stats stats = LeafletLevelN(5);

		ExpectEqual(stats, {
			.totalOpaque = 207,
			.totalTransparent = 554,
			.totalUnknownTransparent = 139,
			.totalUnknownOpaque = 124,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel6) {

		omm::Debug::Stats stats = LeafletLevelN(6);

		ExpectEqual(stats, {
			.totalOpaque = 1021,
			.totalTransparent = 2508,
			.totalUnknownTransparent = 275,
			.totalUnknownOpaque = 292,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel7) {

		omm::Debug::Stats stats = LeafletLevelN(7);

		ExpectEqual(stats, {
			.totalOpaque = 4666,
			.totalTransparent = 10580,
			.totalUnknownTransparent = 549,
			.totalUnknownOpaque = 589,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel8) {

		omm::Debug::Stats stats = LeafletLevelN(8);

		ExpectEqual(stats, {
			.totalOpaque = 19831,
			.totalTransparent = 43424,
			.totalUnknownTransparent = 1110,
			.totalUnknownOpaque = 1171,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel12_TooBigWorkload) {

		omm::Debug::Stats stats = LeafletLevelN(12, 512, omm::Result::WORKLOAD_TOO_BIG);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 0,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 0,
			});
	}

	TEST_P(OMMBakeTestCPU, DeserializeInput_v1_4_0) {

		// GenerateSerializedString("input_v1_4_0", nullptr, false);

		std::vector<unsigned char> input_v1_4_0 
			= { 0x7B, 0xF1, 0x3F, 0x42, 0x79, 0xB1, 0x1D, 0x05, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
				0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x9A, 0x99, 0x19, 0x3F,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
				0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F,
				0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
				0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00 };

		omm::Debug::Stats stats = Bake(input_v1_4_0);

		ExpectEqual(stats, {
			.totalOpaque = 152,
			.totalTransparent = 232,
			.totalUnknownTransparent = 70,
			.totalUnknownOpaque = 58,
			});
	}

	TEST_P(OMMBakeTestCPU, DeserializeOutput_v1_4_0) {

		// GenerateSerializedString(nullptr, "output_v1_4_0", false);

		std::vector<unsigned char> output_v1_4_0 
			= { 0x0E, 0x73, 0xCB, 0x81, 0x3B, 0xF6, 0xA4, 0xA1, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
				0x55, 0x55, 0x75, 0xD5, 0xEF, 0xAB, 0x00, 0xFE, 0x8A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0xA8, 0xDE, 0x00, 0x00, 0x2A, 0xA8, 0x7F,
				0x55, 0x55, 0x55, 0xFD, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A, 0xA8, 0xDF, 0x00, 0x00, 0x2A, 0x80, 0x7E,
				0x55, 0x55, 0x7F, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
				0xED, 0xAB, 0x02, 0xFE, 0xAB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xBA, 0xFE, 0x6F, 0xD5, 0x6F, 0xD5, 0x2A,
				0x80, 0x2A, 0xA8, 0x7F, 0x55, 0x55, 0x55, 0xFD, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A, 0xA8, 0xDF, 0x02,
				0xA8, 0xDE, 0x57, 0x75, 0x55, 0x55, 0x7F, 0x55, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x02, 0x00,
				0x40, 0x00, 0x00, 0x00, 0x04, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x02, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
				0x04, 0x00, 0x02, 0x00 };

		omm::Debug::Stats stats = GetBakeOutput(output_v1_4_0);

		ExpectEqual(stats, {
			.totalOpaque = 152,
			.totalTransparent = 232,
			.totalUnknownTransparent = 70,
			.totalUnknownOpaque = 58,
			});
	}

	TEST_P(OMMBakeTestCPU, DeserializeOutput_Compress_v1_4_0) {

		// GenerateSerializedString(nullptr, "output_compress_v1_4_0", true);

		std::vector<unsigned char> output_compress_v1_4_0
			= { 0x0E, 0x73, 0xCB, 0x81, 0x3B, 0xF6, 0xA4, 0xA1, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
				0x55, 0x55, 0x75, 0xD5, 0xEF, 0xAB, 0x00, 0xFE, 0x8A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0xA8, 0xDE, 0x00, 0x00, 0x2A, 0xA8, 0x7F,
				0x55, 0x55, 0x55, 0xFD, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A, 0xA8, 0xDF, 0x00, 0x00, 0x2A, 0x80, 0x7E,
				0x55, 0x55, 0x7F, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
				0xED, 0xAB, 0x02, 0xFE, 0xAB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xBA, 0xFE, 0x6F, 0xD5, 0x6F, 0xD5, 0x2A,
				0x80, 0x2A, 0xA8, 0x7F, 0x55, 0x55, 0x55, 0xFD, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A, 0xA8, 0xDF, 0x02,
				0xA8, 0xDE, 0x57, 0x75, 0x55, 0x55, 0x7F, 0x55, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x02, 0x00,
				0x40, 0x00, 0x00, 0x00, 0x04, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x02, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
				0x04, 0x00, 0x02, 0x00 };

		omm::Debug::Stats stats = GetBakeOutput(output_compress_v1_4_0);

		ExpectEqual(stats, {
			.totalOpaque = 152,
			.totalTransparent = 232,
			.totalUnknownTransparent = 70,
			.totalUnknownOpaque = 58,
			});
	}

	TEST_P(OMMBakeTestCPU, DeserializeInput_v1_5_0) {

		// GenerateSerializedString("input_v1_5_0", nullptr, false);

		std::vector<unsigned char> input_v1_5_0 
			= { 0xF9, 0x28, 0x27, 0x49, 0x16, 0xE8, 0xF3, 0x2F, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
				0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x3E,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x9A, 0x99, 0x19, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x02, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F,
				0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
				0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFC, 0xFF, 0xFF, 0xFF,
				0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00,
				0x00 };

		omm::Debug::Stats stats = Bake(input_v1_5_0);

		ExpectEqual(stats, {
			.totalOpaque = 152,
			.totalTransparent = 232,
			.totalUnknownTransparent = 70,
			.totalUnknownOpaque = 58,
			});
	}

	TEST_P(OMMBakeTestCPU, DeserializeInput_Compress_v1_5_0) {

		// GenerateSerializedString("input_compress_v1_5_0", nullptr, true);

		std::vector<unsigned char> input_compress_v1_5_0 
			= { 0xCC, 0xA7, 0x2B, 0xC2, 0xF9, 0x90, 0x5B, 0x10, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xE9, 0x01, 0x00, 0x00, 0x44, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x13,
				0x08, 0x04, 0x00, 0x43, 0x00, 0x00, 0x00, 0x3E, 0x04, 0x00, 0x01, 0x02, 0x00, 0x11, 0x40, 0x06, 0x00, 0x06, 0x02, 0x00,
				0x04, 0x30, 0x00, 0x01, 0x39, 0x00, 0xAF, 0x00, 0x00, 0x00, 0x9A, 0x99, 0x19, 0x3F, 0x00, 0x00, 0x80, 0x04, 0x00, 0x04,
				0x00, 0x02, 0x00, 0x08, 0x1C, 0x00, 0x0F, 0x02, 0x00, 0x01, 0x04, 0x20, 0x00, 0x04, 0x02, 0x00, 0x04, 0x10, 0x00, 0x04,
				0x08, 0x00, 0x0F, 0x02, 0x00, 0x01, 0x00, 0x1C, 0x00, 0x04, 0x02, 0x00, 0x04, 0x0C, 0x00, 0x04, 0x08, 0x00, 0x0C, 0x02,
				0x00, 0x00, 0x18, 0x00, 0x08, 0x04, 0x00, 0x04, 0x02, 0x00, 0x00, 0x14, 0x00, 0x0F, 0x02, 0x00, 0x25, 0x00, 0x3C, 0x00,
				0x04, 0x04, 0x00, 0x02, 0x02, 0x00, 0x16, 0x02, 0x13, 0x01, 0x01, 0x02, 0x00, 0x00, 0x10, 0x00, 0x11, 0x20, 0x0A, 0x00,
				0x0C, 0x02, 0x00, 0x08, 0x38, 0x00, 0x06, 0x0C, 0x00, 0x00, 0x58, 0x01, 0x13, 0x06, 0x2B, 0x00, 0x00, 0x0C, 0x00, 0x00,
				0x40, 0x00, 0x13, 0x03, 0x54, 0x00, 0x00, 0x0C, 0x00, 0x07, 0x02, 0x00, 0x10, 0x3F, 0x0C, 0x00, 0x04, 0x24, 0x00, 0x08,
				0x02, 0x00, 0x54, 0xFC, 0xFF, 0xFF, 0xFF, 0x04, 0x11, 0x00, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00,
				0x00, 0x00, 0x00 };

		omm::Debug::Stats stats = Bake(input_compress_v1_5_0);

		ExpectEqual(stats, {
			.totalOpaque = 152,
			.totalTransparent = 232,
			.totalUnknownTransparent = 70,
			.totalUnknownOpaque = 58,
			});
	}

	TEST_P(OMMBakeTestCPU, DeserializeOutput_v1_5_0) {

		// GenerateSerializedString(nullptr, "output_v1_5_0", false);

		std::vector<unsigned char> output_v1_5_0 
			= { 0xEC, 0x20, 0xE6, 0x60, 0x0F, 0x57, 0xC1, 0x93, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
				0x80, 0x00, 0x00, 0x00, 0x55, 0x55, 0x75, 0xD5, 0xEF, 0xAB, 0x00, 0xFE, 0x8A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0xA8, 0xDE, 0x00,
				0x00, 0x2A, 0xA8, 0x7F, 0x55, 0x55, 0x55, 0xFD, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A, 0xA8, 0xDF, 0x00,
				0x00, 0x2A, 0x80, 0x7E, 0x55, 0x55, 0x7F, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
				0x55, 0x55, 0x55, 0x55, 0xED, 0xAB, 0x02, 0xFE, 0xAB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xBA, 0xFE, 0x6F,
				0xD5, 0x6F, 0xD5, 0x2A, 0x80, 0x2A, 0xA8, 0x7F, 0x55, 0x55, 0x55, 0xFD, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x2A, 0xA8, 0xDF, 0x02, 0xA8, 0xDE, 0x57, 0x75, 0x55, 0x55, 0x7F, 0x55, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x04, 0x00, 0x02, 0x00, 0x40, 0x00, 0x00, 0x00, 0x04, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
				0x04, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
				0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x02, 0x00 };

		omm::Debug::Stats stats = GetBakeOutput(output_v1_5_0);

		ExpectEqual(stats, {
			.totalOpaque = 152,
			.totalTransparent = 232,
			.totalUnknownTransparent = 70,
			.totalUnknownOpaque = 58,
			});
	}

	TEST_P(OMMBakeTestCPU, DeserializeOutput_Compress_v1_5_0) {

		// GenerateSerializedString(nullptr, "output_compress_v1_5_0", true);

		std::vector<unsigned char> output_compress_v1_5_0 
			= { 0x33, 0x2C, 0x5C, 0x0E, 0xFB, 0x84, 0xDA, 0xE1, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xC4, 0x00, 0x00, 0x00, 0xF0, 0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
				0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x55, 0x55, 0x75, 0xD5, 0xEF, 0xAB, 0x00, 0xFE, 0x8A, 0x15, 0x00, 0x21, 0x00, 0x2A,
				0x06, 0x00, 0x08, 0x02, 0x00, 0xC4, 0x20, 0xA8, 0xDE, 0x00, 0x00, 0x2A, 0xA8, 0x7F, 0x55, 0x55, 0x55, 0xFD, 0x18, 0x00,
				0xA0, 0x2A, 0xA8, 0xDF, 0x00, 0x00, 0x2A, 0x80, 0x7E, 0x55, 0x55, 0x17, 0x00, 0x0A, 0x02, 0x00, 0x53, 0xED, 0xAB, 0x02,
				0xFE, 0xAB, 0x29, 0x00, 0x9E, 0x80, 0xBA, 0xFE, 0x6F, 0xD5, 0x6F, 0xD5, 0x2A, 0x80, 0x40, 0x00, 0x50, 0x02, 0xA8, 0xDE,
				0x57, 0x75, 0x40, 0x00, 0x13, 0x02, 0x2C, 0x00, 0x53, 0x04, 0x00, 0x02, 0x00, 0x40, 0x08, 0x00, 0x00, 0x9C, 0x00, 0x00,
				0x18, 0x00, 0x00, 0x14, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x06, 0x00, 0x00, 0x14, 0x00, 0x00, 0x04, 0x00, 0x80, 0x02, 0x00,
				0x00, 0x00, 0x04, 0x00, 0x02, 0x00 };

		omm::Debug::Stats stats = GetBakeOutput(output_compress_v1_5_0);

		ExpectEqual(stats, {
			.totalOpaque = 152,
			.totalTransparent = 232,
			.totalUnknownTransparent = 70,
			.totalUnknownOpaque = 58,
			});
	}

	TEST_P(OMMBakeTestCPU, DeserializeOutput_Compress_v1_6_0) {

		// GenerateSerializedString(nullptr, "output_compress_v1_6_0", true);

		std::vector<unsigned char> output_compress_v1_6_0 
			= { 0xAF, 0xDA, 0x05, 0x53, 0xD2, 0xEA, 0xA3, 0x8D, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xC4, 0x00, 0x00, 0x00, 0xF0, 0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
				0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x55, 0x55, 0x75, 0xD5, 0xEF, 0xAB, 0x00, 0xFE, 0x8A, 0x15, 0x00, 0x21, 0x00, 0x2A,
				0x06, 0x00, 0x08, 0x02, 0x00, 0xC4, 0x20, 0xA8, 0xDE, 0x00, 0x00, 0x2A, 0xA8, 0x7F, 0x55, 0x55, 0x55, 0xFD, 0x18, 0x00,
				0xA0, 0x2A, 0xA8, 0xDF, 0x00, 0x00, 0x2A, 0x80, 0x7E, 0x55, 0x55, 0x17, 0x00, 0x0A, 0x02, 0x00, 0x53, 0xED, 0xAB, 0x02,
				0xFE, 0xAB, 0x29, 0x00, 0x9E, 0x80, 0xBA, 0xFE, 0x6F, 0xD5, 0x6F, 0xD5, 0x2A, 0x80, 0x40, 0x00, 0x50, 0x02, 0xA8, 0xDE,
				0x57, 0x75, 0x40, 0x00, 0x13, 0x02, 0x2C, 0x00, 0x53, 0x04, 0x00, 0x02, 0x00, 0x40, 0x08, 0x00, 0x00, 0x9C, 0x00, 0x00,
				0x18, 0x00, 0x00, 0x14, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x06, 0x00, 0x00, 0x14, 0x00, 0x00, 0x04, 0x00, 0x80, 0x02, 0x00,
				0x00, 0x00, 0x04, 0x00, 0x02, 0x00 };

		omm::Debug::Stats stats = GetBakeOutput(output_compress_v1_6_0);

		ExpectEqual(stats, {
			.totalOpaque = 152,
			.totalTransparent = 232,
			.totalUnknownTransparent = 70,
			.totalUnknownOpaque = 58,
			});
	}

	TEST_P(OMMBakeTestCPU, DeserializeOutput_Compress_v1_7_0) {

		// GenerateSerializedString(nullptr, "output_compress_v1_7_0", true);

		std::vector<unsigned char> output_compress_v1_7_0 
			= { 0xE3, 0x76, 0x15, 0xC4, 0x57, 0x41, 0x7D, 0x25, 0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xC4, 0x00, 0x00, 0x00, 0xF0, 0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
				0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x55, 0x55, 0x75, 0xD5, 0xEF, 0xAB, 0x00, 0xFE, 0x8A, 0x15, 0x00, 0x21, 0x00, 0x2A,
				0x06, 0x00, 0x08, 0x02, 0x00, 0xC4, 0x20, 0xA8, 0xDE, 0x00, 0x00, 0x2A, 0xA8, 0x7F, 0x55, 0x55, 0x55, 0xFD, 0x18, 0x00,
				0xA0, 0x2A, 0xA8, 0xDF, 0x00, 0x00, 0x2A, 0x80, 0x7E, 0x55, 0x55, 0x17, 0x00, 0x0A, 0x02, 0x00, 0x53, 0xED, 0xAB, 0x02,
				0xFE, 0xAB, 0x29, 0x00, 0x9E, 0x80, 0xBA, 0xFE, 0x6F, 0xD5, 0x6F, 0xD5, 0x2A, 0x80, 0x40, 0x00, 0x50, 0x02, 0xA8, 0xDE,
				0x57, 0x75, 0x40, 0x00, 0x13, 0x02, 0x2C, 0x00, 0x53, 0x04, 0x00, 0x02, 0x00, 0x40, 0x08, 0x00, 0x00, 0x9C, 0x00, 0x00,
				0x18, 0x00, 0x00, 0x14, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x06, 0x00, 0x00, 0x14, 0x00, 0x00, 0x04, 0x00, 0x80, 0x02, 0x00,
				0x00, 0x00, 0x04, 0x00, 0x02, 0x00 };

		omm::Debug::Stats stats = GetBakeOutput(output_compress_v1_7_0);

		ExpectEqual(stats, {
			.totalOpaque = 152,
			.totalTransparent = 232,
			.totalUnknownTransparent = 70,
			.totalUnknownOpaque = 58,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_lvl1) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,
								0.2f, 0.437582970f,
								0.2f, /*0.221271083f*/ 0.218791485f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 1, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle);

		ExpectEqual(stats, {
			.totalOpaque = 1,
			.totalUnknownTransparent = 1,
			.totalUnknownOpaque = 2,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_lvl2) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,
								0.2f, 0.437582970f,
								0.2f, /*0.221271083f*/ 0.218791485f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 2, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle);

		ExpectEqual(stats, {
			.totalOpaque = 6,
			.totalTransparent = 3,
			.totalUnknownTransparent = 3,
			.totalUnknownOpaque = 4,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_Horizontal) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.2f,
								0.3f, 0.2f,
								0.41f, 0.2f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 1, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 3,
			.totalUnknownTransparent = 1,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_Diagonal) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.2f,
								0.3f, 0.2f,
								0.4f, 0.2f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 2, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle);

		ExpectEqual(stats, {
			.totalTransparent = 13,
			.totalUnknownTransparent = 2,
			.totalUnknownOpaque = 1,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_lvl3) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,
								0.2f, 0.437582970f,
								0.2f, /*0.221271083f*/ 0.218791485f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 3, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle);

		ExpectEqual(stats, {
			.totalOpaque = 28,
			.totalTransparent = 21,
			.totalUnknownTransparent = 7,
			.totalUnknownOpaque = 8,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_lvl4) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,
								0.2f, 0.437582970f,
								0.2f, /*0.221271083f*/ 0.218791485f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 4, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle);

		ExpectEqual(stats, {
			.totalOpaque = 136,
			.totalTransparent = 91,
			.totalUnknownTransparent = 14,
			.totalUnknownOpaque = 15,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_lvl4_wrap) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = {  -0.8f, 0.f,
								-0.8f, 0.437582970f,
								-0.8f, /*0.221271083f*/ 0.218791485f };
		
		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 4, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle, 
			{ .addressingMode = omm::TextureAddressMode::Wrap, .oneFile = false, .detailedCutout = true });

		ExpectEqual(stats, {
			.totalOpaque = 136,
			.totalTransparent = 91,
			.totalUnknownTransparent = 14,
			.totalUnknownOpaque = 15,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_dyn_lvl_0_1) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,
								0.2f, 0.437582970f,
								0.2f, /*0.221271083f*/ 0.218791485f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 12, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle, { .dynamicSubdivisionScale = 0.1f });

		ExpectEqual(stats, {
			.totalOpaque = 9642463,
			.totalTransparent = 7108335,
			.totalUnknownTransparent = 3771,
			.totalUnknownOpaque = 22647,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_dyn_lvl_0_5) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,
								0.2f, 0.437582970f,
								0.2f, /*0.221271083f*/ 0.218791485f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 12, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle, { .dynamicSubdivisionScale = 0.5f });

		ExpectEqual(stats, {
			.totalOpaque = 601591,
			.totalTransparent = 443211,
			.totalUnknownTransparent = 942,
			.totalUnknownOpaque = 2832,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_dyn_lvl_2) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = {  0.2f, 0.f,
								0.2f, 0.437582970f,
								0.2f, /*0.221271083f*/ 0.218791485f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 12, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle, {.dynamicSubdivisionScale = 2.f});

		ExpectEqual(stats, {
			.totalOpaque = 37333,
			.totalTransparent = 27495,
			.totalUnknownTransparent = 353,
			.totalUnknownOpaque = 355,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_dyn_lvl_3) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,
								0.2f, 0.437582970f,
								0.2f, /*0.221271083f*/ 0.218791485f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 12, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle, { .dynamicSubdivisionScale = 3.f });

		ExpectEqual(stats, {
			.totalOpaque = 37333,
			.totalTransparent = 27495,
			.totalUnknownTransparent = 353,
			.totalUnknownOpaque = 355,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Default_dyn_lvl_10) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,
								0.2f, 0.437582970f,
								0.2f, /*0.221271083f*/ 0.218791485f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 12, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle, { .dynamicSubdivisionScale = 10.f });

		ExpectEqual(stats, {
			.totalOpaque = 2266,
			.totalTransparent = 1653,
			.totalUnknownTransparent = 87,
			.totalUnknownOpaque = 90,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Point_Transparent) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.437582970f,
								0.2f, 0.437582970f,
								0.2f, 0.437582970f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 12, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle, { .dynamicSubdivisionScale = 2.f });

		ExpectEqual(stats, {
			.totalFullyTransparent = 1,
			});
	}

	TEST_P(OMMBakeTestCPU, Degen_Point_Opaque) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.1f,
								0.2f, 0.1f,
								0.2f, 0.1f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 12, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords, &StandardCircle, { .dynamicSubdivisionScale = 2.f });

		ExpectEqual(stats, {
			.totalFullyOpaque = 1,
			});
	}

	TEST_P(OMMBakeTestCPU, Invalid_FullyUnknownTransparent) {

		uint32_t triangleIndices[3] = { 0, 1, 2, };
		float texCoords[8] = { 0.f, 0.f,
								0.f, std::numeric_limits<float>::quiet_NaN(),
								0.f, 0.221271083f };

		omm::Debug::Stats stats = GetOmmBakeStatsFP32(0.5f, 4, { 1024, 1024 }, 3, triangleIndices, omm::TexCoordFormat::UV32_FLOAT, texCoords,
			&StandardCircle, { .unresolvedTriState = omm::SpecialIndex::FullyUnknownTransparent });

		ExpectEqual(stats, {
			.totalFullyUnknownTransparent = 1,
			});
	}

	TEST_P(OMMBakeTestCPU, DestroyOpacityMicromapBaker) {
	}

	TEST(VMUtil, GetNumMicroTriangles) {
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(0), 1);
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(1), 4);
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(2), 16);
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(3), 64);
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(4), 256);
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(5), 1024);
	}

	std::string CustomParamName(const ::testing::TestParamInfo<TestSuiteConfig>& info) {

		std::string str = "";
		if ((info.param & TestSuiteConfig::Default) == TestSuiteConfig::Default)
			str += "Default_";
		if ((info.param & TestSuiteConfig::TextureDisableZOrder) == TestSuiteConfig::TextureDisableZOrder)
			str += "TextureDisableZOrder_";
		if ((info.param & TestSuiteConfig::Force32BitIndices) == TestSuiteConfig::Force32BitIndices)
			str += "Force32BitIndices_";
		if ((info.param & TestSuiteConfig::TextureAsUNORM8) == TestSuiteConfig::TextureAsUNORM8)
			str += "TextureAsUNORM8_";
		if ((info.param & TestSuiteConfig::AlphaCutoff) == TestSuiteConfig::AlphaCutoff)
			str += "AlphaCutoff_";
		if (str.length() > 0)
			str.pop_back();
		return str;
	}

	INSTANTIATE_TEST_SUITE_P(OMMTestCPU, OMMBakeTestCPU, ::testing::Values(
		   TestSuiteConfig::Default
		 , TestSuiteConfig::TextureDisableZOrder
		 , TestSuiteConfig::Force32BitIndices
		 , TestSuiteConfig::TextureAsUNORM8
		 , TestSuiteConfig::AlphaCutoff
		 , TestSuiteConfig::Serialize
		
	), CustomParamName);

}  // namespace