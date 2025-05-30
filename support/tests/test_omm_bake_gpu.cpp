/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>

#include "gtest/gtest.h"
#include "util/omm_histogram.h"
#include "nvrhi/nvrhi_environment.h"
#include "nvrhi/nvrhi_wrapper.h"
#include "nvrhi/nvrhi_environment.h"

#include <nvrhi/nvrhi.h>

#include <omm-gpu-nvrhi.h>
#include <omm.h>
#include "util/bird.h"

namespace {

	enum TestSuiteConfig
	{
		None							= 0,
		ComputeOnly						= 1u << 0,
		DisableSpecialIndices			= 1u << 1,
		Force32BitIndices				= 1u << 2,
		DisableTexCoordDeduplication	= 1u << 3,
		RedChannel						= 1u << 4,
		GreenChannel					= 1u << 5,
		BlueChannel						= 1u << 6,
		SetupBeforeBuild				= 1u << 7,
		EnablePostDispatchInfoStats		= 1u << 8,
	};

	class OMMBakeTestGPU : public ::testing::TestWithParam<TestSuiteConfig> {
	protected:
		void SetUp() override {
			ASSERT_NE(g_nvrhiEnvironment, nullptr);
			NVRHIContext* nvrhiContext = g_nvrhiEnvironment->GetContext();
			ASSERT_NE(nvrhiContext, nullptr);
			m_device = nvrhiContext->CreateDevice();
			ASSERT_NE(m_device.Get(), nullptr);

			m_commandList = m_device->createCommandList();
			ASSERT_NE(m_commandList.Get(), nullptr);
		}

		void TearDown() override {
			m_commandList = nullptr;
			m_device = nullptr;
		}

		bool SetupBeforeBuild() const { return (GetParam() & TestSuiteConfig::SetupBeforeBuild) == TestSuiteConfig::SetupBeforeBuild; }
		bool ComputeOnly() const { return (GetParam() & TestSuiteConfig::ComputeOnly) == TestSuiteConfig::ComputeOnly; }
		bool EnablePostDispatchInfoStats() const { return (GetParam() & TestSuiteConfig::EnablePostDispatchInfoStats) == TestSuiteConfig::EnablePostDispatchInfoStats; }
		bool EnableSpecialIndices() const { return (GetParam() & TestSuiteConfig::DisableSpecialIndices) != TestSuiteConfig::DisableSpecialIndices; }
		bool Force32BitIndices() const { return (GetParam() & TestSuiteConfig::Force32BitIndices) == TestSuiteConfig::Force32BitIndices; }
		bool EnableTexCoordDeduplication() const { return (GetParam() & TestSuiteConfig::DisableTexCoordDeduplication) != TestSuiteConfig::DisableTexCoordDeduplication; }
		uint GetAlphaChannelIndex() const { 
			if ((GetParam() & TestSuiteConfig::RedChannel) == TestSuiteConfig::RedChannel)
				return 0;
			if ((GetParam() & TestSuiteConfig::GreenChannel) == TestSuiteConfig::GreenChannel)
				return 1;
			if ((GetParam() & TestSuiteConfig::BlueChannel) == TestSuiteConfig::BlueChannel)
				return 2;
			return 3;
		}

		std::vector<uint32_t> ConvertTexCoords(nvrhi::Format format, float* texCoords, uint32_t texCoordsSize)
		{
			if (format == nvrhi::Format::R16_FLOAT || format == nvrhi::Format::R16_UNORM)
			{
				uint32_t numTexCoordPairs = texCoordsSize / sizeof(float2);
				std::vector<uint32_t> texCoordData(numTexCoordPairs);
				for (uint i = 0; i < numTexCoordPairs; ++i)
				{
					glm::vec2 v;
					v.x = ((float*)texCoords)[2 * i + 0];
					v.y = ((float*)texCoords)[2 * i + 1];

					if (format == nvrhi::Format::R16_UNORM)
						texCoordData[i] = glm::packUnorm2x16(v);
					else if (format == nvrhi::Format::R16_FLOAT)
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

		struct OmmBakeParams
		{
			float alphaCutoff = 0.5f;
			omm::OpacityState alphaCutoffGT = omm::OpacityState::Opaque;
			omm::OpacityState alphaCutoffLE = omm::OpacityState::Transparent;
			uint32_t subdivisionLevel = 5;
			int2 texSize = { 1024, 1024 };
			uint32_t indexBufferSize = 0;
			uint32_t* triangleIndices = nullptr;
			nvrhi::Format texCoordFormat = nvrhi::Format::R32_FLOAT;
			void* texCoords = nullptr;
			uint32_t texCoordBufferSize = 0;
			uint32_t maxOutOmmArraySize = 0xFFFFFFFF;
			std::function<float(int i, int j)> texCb;
			omm::Format format = omm::Format::OC1_4_State;

			static OmmBakeParams InitQuad()
			{
				OmmBakeParams p;
				static uint32_t s_triangleIndices[] = { 0, 1, 2, 3, 1, 2 };
				static float s_texCoords[] = { 0.f, 0.f,	0.f, 1.f,	1.f, 0.f,	 1.f, 1.f };
				p.triangleIndices = s_triangleIndices;
				p.indexBufferSize = sizeof(s_triangleIndices);
				p.texCoords = s_texCoords;
				p.texCoordBufferSize = sizeof(s_texCoords);
				return p;
			}
		};

		omm::Debug::Stats RunOmmBake(const OmmBakeParams& p) 
		{
			const uint32_t alphaTextureChannel = GetAlphaChannelIndex();

			nvrhi::TextureDesc desc;
			desc.width = p.texSize.x;
			desc.height = p.texSize.y;
			desc.format = nvrhi::Format::RGBA32_FLOAT;

			nvrhi::StagingTextureHandle staging = m_device->createStagingTexture(desc, nvrhi::CpuAccessMode::Write);

			nvrhi::TextureSlice slice;
			slice = slice.resolve(desc);
			size_t rowPitch;
			void* data = m_device->mapStagingTexture(staging.Get(), slice, nvrhi::CpuAccessMode::Write, &rowPitch);
			std::vector<float> imageData;
			for (uint32_t j = 0; j < desc.height; ++j)
			{
				for (uint32_t i = 0; i < desc.width; ++i)
				{
					float* rgba = (float*)((uint8_t*)data + j * rowPitch + (4 * i) * sizeof(float));
					float val = p.texCb(i, j);
					rgba[0] = alphaTextureChannel == 0 ? val : 0.f;
					rgba[1] = alphaTextureChannel == 1 ? val : 0.f;
					rgba[2] = alphaTextureChannel == 2 ? val : 0.f;
					rgba[3] = alphaTextureChannel == 3 ? val : 0.f;
					imageData.push_back(val);
				}
			}

			m_device->unmapStagingTexture(staging.Get());

			// Upload alpha texture
			m_commandList->open();
			nvrhi::TextureHandle alphaTexture;
			{
				alphaTexture = m_device->createTexture(desc);
				m_commandList->beginTrackingTextureState(alphaTexture, nvrhi::TextureSubresourceSet(), nvrhi::ResourceStates::Common);
				m_commandList->copyTexture(alphaTexture, slice, staging, slice);
			}

			// Upload index buffer
			nvrhi::BufferHandle ib;
			{
				ib = m_device->createBuffer({ .byteSize = p.indexBufferSize, .debugName = "ib", .format = nvrhi::Format::R32_UINT, .canHaveUAVs = true, .canHaveTypedViews = true, .canHaveRawViews = true });
				m_commandList->beginTrackingBufferState(ib, nvrhi::ResourceStates::Common);
				m_commandList->writeBuffer(ib, p.triangleIndices, p.indexBufferSize);
			}

			// Upload texcoords
			nvrhi::BufferHandle vb;
			{
				vb = m_device->createBuffer({ .byteSize = p.texCoordBufferSize, .debugName = "vb", .canHaveUAVs = true, .canHaveRawViews = true });
				m_commandList->beginTrackingBufferState(vb, nvrhi::ResourceStates::Common);

				m_commandList->writeBuffer(vb, p.texCoords, p.texCoordBufferSize);
			}

			// Upload index buffer
			omm::GpuBakeNvrhi bake(m_device, m_commandList, true /*enable debug*/);

			omm::GpuBakeNvrhi::Input input;
			input.alphaTexture = alphaTexture;
			input.alphaTextureChannel = alphaTextureChannel;
			input.alphaCutoff = p.alphaCutoff;
			input.alphaCutoffLessEqual = p.alphaCutoffLE;
			input.alphaCutoffGreater = p.alphaCutoffGT;
			input.texCoordFormat = p.texCoordFormat;
			input.texCoordBuffer = vb;
			input.texCoordStrideInBytes = 0;
			input.indexBuffer = ib;
			input.numIndices = p.indexBufferSize / sizeof(uint32_t);
			input.maxSubdivisionLevel = p.subdivisionLevel;
			input.format = p.format == omm::Format::OC1_2_State ? nvrhi::rt::OpacityMicromapFormat::OC1_2_State : nvrhi::rt::OpacityMicromapFormat::OC1_4_State;
			input.dynamicSubdivisionScale = 0.f;
			input.enableStats = EnablePostDispatchInfoStats();
			input.enableSpecialIndices = EnableSpecialIndices();
			input.force32BitIndices = Force32BitIndices();
			input.enableTexCoordDeduplication = EnableTexCoordDeduplication();
			input.computeOnly = ComputeOnly();
			input.maxOutOmmArraySize = p.maxOutOmmArraySize;

			// Readback.
			auto ReadBuffer = [this](nvrhi::BufferHandle buffer, size_t size = 0xFFFFFFFF)->std::vector<uint8_t>
			{
				if (size == 0)
					return {};
				std::vector<uint8_t> data;
				void* pData = m_device->mapBuffer(buffer, nvrhi::CpuAccessMode::Read);
				assert(pData);
				size_t byteSize = size == 0xFFFFFFFF ? buffer->getDesc().byteSize : size;
				assert(byteSize <= buffer->getDesc().byteSize);
				data.resize(byteSize);
				memcpy(data.data(), pData, byteSize);
				m_device->unmapBuffer(buffer);
				return data;
			};

			omm::GpuBakeNvrhi::Buffers res;
			nvrhi::Format ommIndexFormat = nvrhi::Format::UNKNOWN;
			uint32_t ommIndexCount = 0xFFFFFFFF;
			if (SetupBeforeBuild())
			{
				input.operation = omm::GpuBakeNvrhi::Operation::Setup;

				omm::GpuBakeNvrhi::PreDispatchInfo info;
				bake.GetPreDispatchInfo(input, info);
				ommIndexFormat = info.ommIndexFormat;
				ommIndexCount = info.ommIndexCount;

				res.ommDescBuffer = m_device->createBuffer({ .byteSize = info.ommDescBufferSize, .debugName = "omDescBuffer", .canHaveUAVs = true, .canHaveRawViews = true });
				res.ommIndexBuffer = m_device->createBuffer({ .byteSize = info.ommIndexBufferSize, .debugName = "omIndexBuffer", .canHaveUAVs = true, .canHaveRawViews = true });
				res.ommDescArrayHistogramBuffer = m_device->createBuffer({ .byteSize = info.ommDescArrayHistogramSize , .debugName = "omUsageDescBuffer" , .canHaveUAVs = true, .canHaveRawViews = true });
				res.ommIndexHistogramBuffer = m_device->createBuffer({ .byteSize = info.ommIndexHistogramSize , .debugName = "ommIndexHistogramBuffer" , .canHaveUAVs = true, .canHaveRawViews = true });
				res.ommPostDispatchInfoBuffer = m_device->createBuffer({ .byteSize = info.ommPostDispatchInfoBufferSize , .debugName = "ommPostDispatchInfoBuffer" , .canHaveUAVs = true, .canHaveRawViews = true });

				m_commandList->beginTrackingBufferState(res.ommDescBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommIndexBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommDescArrayHistogramBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommIndexHistogramBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommPostDispatchInfoBuffer, nvrhi::ResourceStates::Common);

				omm::GpuBakeNvrhi::Buffers prePass;
				prePass.ommDescBuffer = res.ommDescBuffer;
				prePass.ommIndexBuffer = res.ommIndexBuffer;
				prePass.ommDescArrayHistogramBuffer = res.ommDescArrayHistogramBuffer;
				prePass.ommIndexHistogramBuffer = res.ommIndexHistogramBuffer;
				prePass.ommPostDispatchInfoBuffer = res.ommPostDispatchInfoBuffer;

				bake.Dispatch(
					m_commandList,
					input,
					prePass);

				nvrhi::BufferHandle ommPostDispatchInfoBufferReadback = m_device->createBuffer({ .byteSize = info.ommPostDispatchInfoBufferSize , .debugName = "ommPostDispatchInfoBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
				m_commandList->beginTrackingBufferState(ommPostDispatchInfoBufferReadback, nvrhi::ResourceStates::Common);
				m_commandList->copyBuffer(ommPostDispatchInfoBufferReadback, 0, res.ommPostDispatchInfoBuffer, 0, info.ommPostDispatchInfoBufferSize);

				m_commandList->close();

				// Execute & Sync.
				uint64_t fence = m_device->executeCommandList(m_commandList);

				m_device->waitForIdle();

				std::vector<uint8_t> vmPostDispatchInfoData = ReadBuffer(ommPostDispatchInfoBufferReadback);
				omm::GpuBakeNvrhi::PostDispatchInfo postDispatchInfo;
				omm::GpuBakeNvrhi::ReadPostDispatchInfo(vmPostDispatchInfoData.data(), vmPostDispatchInfoData.size(), postDispatchInfo);

				EXPECT_LE(postDispatchInfo.ommArrayBufferSize, info.ommArrayBufferSize);
				EXPECT_LE(postDispatchInfo.ommDescBufferSize, info.ommDescBufferSize);

				res.ommArrayBuffer = m_device->createBuffer({ .byteSize = std::max(postDispatchInfo.ommArrayBufferSize, 4u), .debugName = "omArrayBuffer", .canHaveUAVs = true, .canHaveRawViews = true});

				m_commandList->open();

				m_commandList->beginTrackingTextureState(alphaTexture, nvrhi::TextureSubresourceSet(), nvrhi::ResourceStates::CopyDest);

				m_commandList->beginTrackingBufferState(ib, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(vb, nvrhi::ResourceStates::Common);

				m_commandList->beginTrackingBufferState(res.ommArrayBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommDescBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommIndexBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommDescArrayHistogramBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommIndexHistogramBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommPostDispatchInfoBuffer, nvrhi::ResourceStates::Common);

				input.operation = omm::GpuBakeNvrhi::Operation::Bake;

				bake.Dispatch(
					m_commandList,
					input,
					res);
			}
			else
			{
				input.operation = omm::GpuBakeNvrhi::Operation::SetupAndBake;

				omm::GpuBakeNvrhi::PreDispatchInfo info;
				bake.GetPreDispatchInfo(input, info);
				ommIndexFormat = info.ommIndexFormat;
				ommIndexCount = info.ommIndexCount;

				res.ommArrayBuffer = m_device->createBuffer({ .byteSize = std::max<size_t>(info.ommArrayBufferSize, 4u), .debugName = "ommArrayBuffer", .canHaveUAVs = true, .canHaveRawViews = true });
				res.ommDescBuffer = m_device->createBuffer({ .byteSize = info.ommDescBufferSize, .debugName = "ommDescBuffer", .canHaveUAVs = true, .canHaveRawViews = true });
				res.ommIndexBuffer = m_device->createBuffer({ .byteSize = info.ommIndexBufferSize, .debugName = "ommIndexBuffer", .canHaveUAVs = true, .canHaveRawViews = true });
				res.ommDescArrayHistogramBuffer = m_device->createBuffer({ .byteSize = info.ommDescArrayHistogramSize , .debugName = "ommUsageDescBuffer" , .canHaveUAVs = true, .canHaveRawViews = true });
				res.ommIndexHistogramBuffer = m_device->createBuffer({ .byteSize = info.ommIndexHistogramSize , .debugName = "ommIndexHistogramBuffer" , .canHaveUAVs = true, .canHaveRawViews = true });
				res.ommPostDispatchInfoBuffer = m_device->createBuffer({ .byteSize = info.ommPostDispatchInfoBufferSize , .debugName = "ommPostDispatchInfoBuffer" , .canHaveUAVs = true, .canHaveRawViews = true });

				m_commandList->beginTrackingBufferState(res.ommArrayBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommDescBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommIndexBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommDescArrayHistogramBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommIndexHistogramBuffer, nvrhi::ResourceStates::Common);
				m_commandList->beginTrackingBufferState(res.ommPostDispatchInfoBuffer, nvrhi::ResourceStates::Common);

				bake.Dispatch(
					m_commandList,
					input,
					res);
			}

			nvrhi::BufferHandle ommArrayBufferReadback = m_device->createBuffer({ .byteSize = res.ommArrayBuffer->getDesc().byteSize, .debugName = "omArrayBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read});
			nvrhi::BufferHandle ommDescBufferReadback = m_device->createBuffer({ .byteSize = res.ommDescBuffer->getDesc().byteSize, .debugName = "omDescBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
			nvrhi::BufferHandle ommIndexBufferReadback = m_device->createBuffer({ .byteSize = res.ommIndexBuffer->getDesc().byteSize, .debugName = "omIndexBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
			nvrhi::BufferHandle ommDescArrayHistogramBufferReadback = m_device->createBuffer({ .byteSize = res.ommDescArrayHistogramBuffer->getDesc().byteSize, .debugName = "vmArrayHistogramBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
			nvrhi::BufferHandle ommIndexHistogramBufferReadback = m_device->createBuffer({ .byteSize = res.ommIndexHistogramBuffer->getDesc().byteSize, .debugName = "vmArrayHistogramBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
			nvrhi::BufferHandle ommPostDispatchInfoBufferReadback = m_device->createBuffer({ .byteSize = res.ommPostDispatchInfoBuffer->getDesc().byteSize, .debugName = "ommPostDispatchInfoBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
			m_commandList->beginTrackingBufferState(ommArrayBufferReadback, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(ommIndexBufferReadback, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(ommDescBufferReadback, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(ommDescArrayHistogramBufferReadback, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(ommIndexHistogramBufferReadback, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(ommPostDispatchInfoBufferReadback, nvrhi::ResourceStates::Common);

			m_commandList->copyBuffer(ommArrayBufferReadback, 0, res.ommArrayBuffer, 0, res.ommArrayBuffer->getDesc().byteSize);
			m_commandList->copyBuffer(ommDescBufferReadback, 0, res.ommDescBuffer, 0, res.ommDescBuffer->getDesc().byteSize);
			m_commandList->copyBuffer(ommIndexBufferReadback, 0, res.ommIndexBuffer, 0, res.ommIndexBuffer->getDesc().byteSize);
			m_commandList->copyBuffer(ommDescArrayHistogramBufferReadback, 0, res.ommDescArrayHistogramBuffer, 0, res.ommDescArrayHistogramBuffer->getDesc().byteSize);
			m_commandList->copyBuffer(ommIndexHistogramBufferReadback, 0, res.ommIndexHistogramBuffer, 0, res.ommIndexHistogramBuffer->getDesc().byteSize);
			m_commandList->copyBuffer(ommPostDispatchInfoBufferReadback, 0, res.ommPostDispatchInfoBuffer, 0, res.ommPostDispatchInfoBuffer->getDesc().byteSize);

			m_commandList->close();

			// Execute & Sync.
			uint64_t fence = m_device->executeCommandList(m_commandList);

			m_device->waitForIdle();

			std::vector<uint8_t> vmPostDispatchInfoData = ReadBuffer(ommPostDispatchInfoBufferReadback);
			omm::GpuBakeNvrhi::PostDispatchInfo postDispatchInfo;
			omm::GpuBakeNvrhi::ReadPostDispatchInfo(vmPostDispatchInfoData.data(), vmPostDispatchInfoData.size(), postDispatchInfo);

			std::vector<uint8_t> ommArrayBufferData = ReadBuffer(ommArrayBufferReadback, postDispatchInfo.ommArrayBufferSize);
			std::vector<uint8_t> ommIndexBufferData = ReadBuffer(ommIndexBufferReadback);
			std::vector<uint8_t> ommDescBufferData = ReadBuffer(ommDescBufferReadback, postDispatchInfo.ommDescBufferSize);
			std::vector<uint8_t> ommArrayHistogramData = ReadBuffer(ommDescArrayHistogramBufferReadback);
			std::vector<uint8_t> ommIndexHistogramData = ReadBuffer(ommIndexHistogramBufferReadback);


			std::string name = ::testing::UnitTest::GetInstance()->current_test_suite()->name();
			std::replace(name.begin(), name.end(), '/', '_');
			std::string tname = ::testing::UnitTest::GetInstance()->current_test_info()->name();
			std::replace(tname.begin(), tname.end(), '/', '_');

#if OMM_TEST_ENABLE_IMAGE_DUMP
			constexpr bool kDumpDebug = true;
#else
			constexpr bool kDumpDebug = false;
#endif

			if constexpr (kDumpDebug)
			{
				bake.DumpDebug(
					"OmmBakeOutput_GPU",
					tname.c_str(),
					input,
					ommArrayBufferData,
					ommDescBufferData,
					ommIndexBufferData,
					ommIndexFormat,
					ommArrayHistogramData,
					ommIndexHistogramData,
					p.triangleIndices,
					p.indexBufferSize / sizeof(uint32_t),
					p.texCoordFormat,
					p.texCoords,
					imageData.data(),
					p.texSize.x,
					p.texSize.y
				);
			}
			size_t indexFormatSize = nvrhi::getFormatInfo(ommIndexFormat).bytesPerBlock;

			omm::Cpu::BakeResultDesc resDesc;
			resDesc.arrayData = ommArrayBufferData.data();
			resDesc.arrayDataSize = (uint32_t)ommArrayBufferData.size();
			resDesc.descArray = (const omm::Cpu::OpacityMicromapDesc*)ommDescBufferData.data();
			resDesc.descArrayCount = (uint32_t)(ommDescBufferData.size() / sizeof(omm::Cpu::OpacityMicromapDesc));
			resDesc.indexBuffer = ommIndexBufferData.data();
			resDesc.indexCount = ommIndexCount;
			if (ommIndexFormat == nvrhi::Format::R8_UINT)
				resDesc.indexFormat = omm::IndexFormat::UINT_8;
			else if (ommIndexFormat == nvrhi::Format::R16_UINT)
				resDesc.indexFormat = omm::IndexFormat::UINT_16;
			else // nvrhi::Format::R32_UINT
				resDesc.indexFormat = omm::IndexFormat::UINT_32;
			resDesc.descArrayHistogram = (const omm::Cpu::OpacityMicromapUsageCount*)ommArrayHistogramData.data();
			resDesc.descArrayHistogramCount = (uint32_t)(ommArrayHistogramData.size() / sizeof(omm::Cpu::OpacityMicromapUsageCount));
			resDesc.indexHistogram = (const omm::Cpu::OpacityMicromapUsageCount*)ommIndexHistogramData.data();
			resDesc.indexHistogramCount = (uint32_t)(ommIndexHistogramData.size() / sizeof(omm::Cpu::OpacityMicromapUsageCount));

			omm::Test::ValidateHistograms(&resDesc);

			omm::GpuBakeNvrhi::Stats stats = bake.GetStats(resDesc);

			if (EnablePostDispatchInfoStats())
			{
				const size_t totalUnknown = stats.totalUnknownOpaque + stats.totalUnknownTransparent;
				const size_t totalFullyUnknown = stats.totalFullyUnknownOpaque + stats.totalFullyUnknownTransparent;
				EXPECT_EQ(postDispatchInfo.ommTotalOpaqueCount, stats.totalOpaque);
				EXPECT_EQ(postDispatchInfo.ommTotalTransparentCount, stats.totalTransparent);
				EXPECT_EQ(postDispatchInfo.ommTotalUnknownCount, totalUnknown);
				EXPECT_EQ(postDispatchInfo.ommTotalFullyOpaqueCount, stats.totalFullyOpaque);
				EXPECT_EQ(postDispatchInfo.ommTotalFullyTransparentCount, stats.totalFullyTransparent);
				EXPECT_EQ(postDispatchInfo.ommTotalFullyUnknownCount, totalFullyUnknown);
			}
			else
			{
				EXPECT_EQ(postDispatchInfo.ommTotalOpaqueCount, 0);
				EXPECT_EQ(postDispatchInfo.ommTotalTransparentCount, 0);
				EXPECT_EQ(postDispatchInfo.ommTotalUnknownCount, 0);
				EXPECT_EQ(postDispatchInfo.ommTotalFullyOpaqueCount, 0);
				EXPECT_EQ(postDispatchInfo.ommTotalFullyTransparentCount, 0);
				EXPECT_EQ(postDispatchInfo.ommTotalFullyUnknownCount, 0);
			}

			return
			{
				.totalOpaque = stats.totalOpaque,
				.totalTransparent = stats.totalTransparent,
				.totalUnknownTransparent = stats.totalUnknownTransparent,
				.totalUnknownOpaque = stats.totalUnknownOpaque,
				.totalFullyOpaque = stats.totalFullyOpaque,
				.totalFullyTransparent = stats.totalFullyTransparent,
				.totalFullyUnknownOpaque = stats.totalFullyUnknownOpaque,
				.totalFullyUnknownTransparent = stats.totalFullyUnknownTransparent
			};
		}

		omm::Debug::Stats RunOmmBake(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			uint32_t indexBufferSize,
			uint32_t* triangleIndices,
			void* texCoords,
			uint32_t texCoordBufferSize,
			std::function<float(int i, int j)> texCb,
			omm::Format format = omm::Format::OC1_4_State,
			nvrhi::Format texCoordFormat = nvrhi::Format::R32_FLOAT)
		{
			OmmBakeParams p;
			p.alphaCutoff = alphaCutoff;
			p.subdivisionLevel = subdivisionLevel;
			p.texSize = texSize;
			p.texCb = texCb;
			p.format = format;
			p.triangleIndices = triangleIndices;
			p.indexBufferSize = indexBufferSize;
			p.texCoordFormat = texCoordFormat;
			p.texCoords = texCoords;
			p.texCoordBufferSize = texCoordBufferSize;
			return RunOmmBake(p);
		}

		omm::Debug::Stats RunOmmBake(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			std::function<float(int i, int j)> tex,
			omm::Format format = omm::Format::OC1_4_State) {
			uint32_t triangleIndices[] = { 0, 1, 2, 3, 1, 2 };
			float texCoords[] = { 0.f, 0.f,	0.f, 1.f,	1.f, 0.f,	 1.f, 1.f };

			OmmBakeParams p;
			p.alphaCutoff = alphaCutoff;
			p.subdivisionLevel = subdivisionLevel;
			p.texSize = texSize;
			p.texCb = tex;
			p.format = format;
			p.triangleIndices = triangleIndices;
			p.indexBufferSize = sizeof(triangleIndices);
			p.texCoords = texCoords;
			p.texCoordBufferSize = sizeof(texCoords);
			return RunOmmBake(p);
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

	private:
		nvrhi::DeviceHandle m_device;
		nvrhi::CommandListHandle m_commandList;
	};


	TEST_P(OMMBakeTestGPU, AllOpaque4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.6f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalOpaque = 512 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllOpaque4_FLIP_T_AND_O) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, 3, 1, 2 };
		float texCoords[] = { 0.f, 0.f,	0.f, 1.f,	1.f, 0.f,	 1.f, 1.f };

		OmmBakeParams p;
		p.alphaCutoff = 0.5f;
		p.alphaCutoffGT = omm::OpacityState::Transparent;
		p.alphaCutoffLE = omm::OpacityState::Opaque;
		p.subdivisionLevel = subdivisionLevel;
		p.texSize = { 1024, 1024 };
		p.texCb = [](int i, int j)->float {
			return 0.6f;
			};
		p.format = omm::Format::OC1_4_State;
		p.triangleIndices = triangleIndices;
		p.indexBufferSize = sizeof(triangleIndices);
		p.texCoords = texCoords;
		p.texCoordBufferSize = sizeof(texCoords);

		omm::Debug::Stats stats = RunOmmBake(p);

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyTransparent = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalTransparent = 512 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllOpaque3) {

		uint32_t subdivisionLevel = 3;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.6f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalOpaque = 128 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllOpaque2) {

		uint32_t subdivisionLevel = 2;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.6f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalOpaque = 32 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllOpaque1) {

		uint32_t subdivisionLevel = 1;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.6f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalOpaque = 8 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllOpaque0) {

		uint32_t subdivisionLevel = 0;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.6f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalOpaque = 2 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparent4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyTransparent = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalTransparent = 512 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparent3) {

		uint32_t subdivisionLevel = 3;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyTransparent = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalTransparent = 128 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparent2) {

		uint32_t subdivisionLevel = 2;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyTransparent = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalTransparent = 32 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparent1) {

		uint32_t subdivisionLevel = 1;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyTransparent = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalTransparent = 8 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparent0) {

		uint32_t subdivisionLevel = 0;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyTransparent = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalTransparent = 2 });
		}
	}

	// A bit lame because GPU baker doesn't differentiate (yet) between UT and UO
	TEST_P(OMMBakeTestGPU, AllUnknownTransparent) {

		uint32_t subdivisionLevel = 1;

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if ((i) % 8 != (j) % 8)
				return 0.f;
			else
				return 1.f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyUnknownOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalUnknownOpaque = 8 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllUnknownOpaque) {

		uint32_t subdivisionLevel = 1;

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if ((i) % 8 != (j) % 8)
				return 1.f;
			else
				return 0.f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyUnknownOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalUnknownOpaque = 8 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparentOpaqueCorner4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, {
				.totalTransparent = numMicroTris - 1,
				.totalUnknownOpaque = 1,
				.totalFullyTransparent = 1,
				});
		}
		else
		{
			ExpectEqual(stats, {
				.totalTransparent = 2 * numMicroTris - 1,
				.totalUnknownOpaque = 1,
				});
		}
	}

	TEST_P(OMMBakeTestGPU, ZeroOmmArraySizeBudget) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		OmmBakeParams p = OmmBakeParams::InitQuad();
		p.subdivisionLevel = 4;
		p.maxOutOmmArraySize = 0;
		p.texCb = [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;
			return 0.4f;
		};

		omm::Debug::Stats stats = RunOmmBake(p);

		ExpectEqual(stats, {
			.totalFullyUnknownOpaque = 2,
			});
	}

	TEST_P(OMMBakeTestGPU, HalfOmmArraySizeBudget) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		OmmBakeParams p = OmmBakeParams::InitQuad();
		p.subdivisionLevel = 4;
		p.maxOutOmmArraySize = 64u; // 64 bytes covers a single subdivlvl 4 prim
		p.texCb = [](int i, int j)->float {
			return 0.4f;
		};

		omm::Debug::Stats stats = RunOmmBake(p);

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { 
				.totalFullyTransparent = 1, 
				.totalFullyUnknownOpaque = 1,  // one triangle is "out of memory"
				});
		}
		else
		{
			ExpectEqual(stats, { 
				.totalTransparent = 256,
				.totalFullyUnknownOpaque = 1 // one triangle is "out of memory"
				});
		}
	}

	TEST_P(OMMBakeTestGPU, Circle) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float r = 0.4f;

			const int2 idx = int2(i, j);
			const float2 uv = float2(idx) / float2((float)1024);
			if (glm::length(uv - 0.5f) < r)
				return 0.f;
			return 1.f;
			});

		ExpectEqual(stats, {
			.totalOpaque = 204,
			.totalTransparent = 219,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 89,
			});
	}

	TEST_P(OMMBakeTestGPU, CircleOC2) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float r = 0.4f;

			const int2 idx = int2(i, j);
			const float2 uv = float2(idx) / float2((float)1024);
			if (glm::length(uv - 0.5f) < r)
				return 0.f;
			return 1.f;
			}, omm::Format::OC1_2_State);

		ExpectEqual(stats, {
			.totalOpaque = 293,
			.totalTransparent = 219,
			});
	}

	TEST_P(OMMBakeTestGPU, Sine) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)1024;

			return 1.f - std::sinf(uv * 15);
			});

		ExpectEqual(stats, {
			.totalOpaque = 224,
			.totalTransparent = 128,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 160,
			});
	}

	TEST_P(OMMBakeTestGPU, SineOC2) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)1024;

			return 1.f - std::sinf(uv * 15);
			}, omm::Format::OC1_2_State);

		ExpectEqual(stats, {
			.totalOpaque = 384,
			.totalTransparent = 128,
			});
	}


	TEST_P(OMMBakeTestGPU, SineOC2Neg) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)1024;

			return 1.f - std::sinf(uv * 15);
			}, omm::Format::OC1_2_State);

		ExpectEqual(stats, {
			.totalOpaque = 384,
			.totalTransparent = 128,
			});
	}

	TEST_P(OMMBakeTestGPU, Mandelbrot) {

		uint32_t subdivisionLevel = 5;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {

			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(1024, 1024) - 0.1f;
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

			}, omm::Format::OC1_4_State);

		ExpectEqual(stats, {
			.totalOpaque = 1212,
			.totalTransparent = 484,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 352,
			});
	}

	TEST_P(OMMBakeTestGPU, Mandelbrot2) {

		uint32_t subdivisionLevel = 5;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, sizeof(triangleIndices), triangleIndices, texCoords, sizeof(texCoords), [](int i, int j)->float {

			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(1024, 1024) - 0.1f;
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

			}, omm::Format::OC1_4_State);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
				.totalOpaque = 522,
				.totalTransparent = 286,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 216,
							});
		}
		else
		{
			ExpectEqual(stats, {
				.totalOpaque = 524,
				.totalTransparent = 287,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 213,
							});
		}
	}

	TEST_P(OMMBakeTestGPU, Mandelbrot3) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, sizeof(triangleIndices), triangleIndices, texCoords, sizeof(texCoords), [](int i, int j)->float {

			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(1024, 1024) - 0.1f;
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

			}, omm::Format::OC1_4_State);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
			.totalOpaque = 164039,
			.totalTransparent = 91321,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 6784,
							});
		}
		else
		{
			ExpectEqual(stats, {
			.totalOpaque = 164027,
			.totalTransparent = 91410,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 6707,
						});
		}
	}

	float GetJulia(int i, int j)
	{
		auto multiply = [](float2 x, float2 y)->float2 {
			return float2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
			};

		float2 uv = 1.2f * float2(i, j) / float2(1024, 1024) - 0.1f;

		float2 z0 = 5.f * (uv - float2(.5, .27));
		float2 col;
		float time = 3.1f;
		float2 c = cos(time) * float2(cos(time / 2.), sin(time / 2.));
		for (int i = 0; i < 500; i++) {
			float2 z = multiply(z0, z0) + c;
			float mq = dot(z, z);
			if (mq > 4.) {
				col = float2(float(i) / 20., 0.);
				break;
			}
			else {
				z0 = z;
			}
			col = float2(mq / 2., mq / 2.);
		}

		float alpha = std::clamp(col.x, 0.f, 1.f);
		return 1.f - alpha;
	}

	TEST_P(OMMBakeTestGPU, Julia) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, sizeof(triangleIndices), triangleIndices, texCoords, sizeof(texCoords), &GetJulia, omm::Format::OC1_4_State);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
				.totalOpaque = 254728,
				.totalTransparent = 4300,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 3116,
				});
		}
		else
		{
			ExpectEqual(stats, {
				.totalOpaque = 254723,
				.totalTransparent = 4300,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 3121,
				});
		}
	}

	TEST_P(OMMBakeTestGPU, Julia_T_AND_UO) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		OmmBakeParams p;
		p.alphaCutoff = 0.5f;
		p.alphaCutoffGT = omm::OpacityState::UnknownOpaque;
		p.alphaCutoffLE = omm::OpacityState::Transparent;
		p.subdivisionLevel = subdivisionLevel;
		p.texSize = { 1024, 1024 };
		p.texCb = &GetJulia;
		p.format = omm::Format::OC1_4_State;
		p.triangleIndices = triangleIndices;
		p.indexBufferSize = sizeof(triangleIndices);
		p.texCoordFormat = nvrhi::Format::R32_FLOAT;
		p.texCoords = texCoords;
		p.texCoordBufferSize = sizeof(texCoords);
		omm::Debug::Stats stats = RunOmmBake(p);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
				.totalOpaque = 0,
				.totalTransparent = 4300,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 3116 + 254728,
				});
		}
		else
		{
			ExpectEqual(stats, {
				.totalOpaque = 0,
				.totalTransparent = 4300,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 3121 + 254723,
				});
		}
	}

	TEST_P(OMMBakeTestGPU, Julia_FLIP_T_AND_O) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		OmmBakeParams p;
		p.alphaCutoff = 0.5f;
		p.alphaCutoffGT = omm::OpacityState::Transparent;
		p.alphaCutoffLE = omm::OpacityState::Opaque;
		p.subdivisionLevel = subdivisionLevel;
		p.texSize = { 1024, 1024 };
		p.texCb = &GetJulia;
		p.format = omm::Format::OC1_4_State;
		p.triangleIndices = triangleIndices;
		p.indexBufferSize = sizeof(triangleIndices);
		p.texCoordFormat = nvrhi::Format::R32_FLOAT;
		p.texCoords = texCoords;
		p.texCoordBufferSize = sizeof(texCoords);
		omm::Debug::Stats stats = RunOmmBake(p);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
				.totalOpaque = 4300,
				.totalTransparent = 254728,
				.totalUnknownTransparent = 3116,
				.totalUnknownOpaque = 0,
				});
		}
		else
		{
			ExpectEqual(stats, {
				.totalOpaque =  4300,
				.totalTransparent = 254723,
				.totalUnknownTransparent = 3121,
				.totalUnknownOpaque = 0,
				});
		}
	}

	TEST_P(OMMBakeTestGPU, Julia_UV_FP16) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		const nvrhi::Format texCoordFormat = nvrhi::Format::R16_FLOAT;
		const std::vector<uint32_t> texCoordFP16 = ConvertTexCoords(texCoordFormat, texCoords, sizeof(texCoords));
		const uint32_t texCoordSize = (uint32_t)(sizeof(uint32_t) * texCoordFP16.size());

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, sizeof(triangleIndices), triangleIndices, (void*)texCoordFP16.data(), texCoordSize, &GetJulia, omm::Format::OC1_4_State, texCoordFormat);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
				.totalOpaque = 254747,
				.totalTransparent = 4304,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 3093,
				});
		}
		else
		{
			ExpectEqual(stats, {
				.totalOpaque = 254746,
				.totalTransparent = 4306,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 3092,
				});
		}
	}

	TEST_P(OMMBakeTestGPU, Julia_UV_UNORM16) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		const nvrhi::Format texCoordFormat = nvrhi::Format::R16_UNORM;
		const std::vector<uint32_t> texCoordUNORM16 = ConvertTexCoords(texCoordFormat, texCoords, sizeof(texCoords));
		const uint32_t texCoordSize = (uint32_t)(sizeof(uint32_t) * texCoordUNORM16.size());

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, sizeof(triangleIndices), triangleIndices, (void*)texCoordUNORM16.data(), texCoordSize, &GetJulia, omm::Format::OC1_4_State, texCoordFormat);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
				.totalOpaque = 254741,
				.totalTransparent = 4312,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 3091,
				});
		}
		else
		{
			ExpectEqual(stats, {
				.totalOpaque = 254737,
				.totalTransparent = 4314,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 3093,
				});
		}
	}

	TEST_P(OMMBakeTestGPU, Julia2x) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, 3, 4, 5, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f, 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 1024, 1024 }, sizeof(triangleIndices), triangleIndices, texCoords, sizeof(texCoords), &GetJulia, omm::Format::OC1_4_State);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
				.totalOpaque = 509456,
				.totalTransparent = 8600,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 6232,
				});
		}
		else
		{
			ExpectEqual(stats, {
				.totalOpaque = 509446,
				.totalTransparent = 8600,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 6242,
				});
		}
	}

	TEST_P(OMMBakeTestGPU, Uniform) {

		uint32_t subdivisionLevel = 6;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, 1, 2, 3, };
		//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
		float texCoords[] = { 0.f, 0.f,  0.f, 1.0f,  1.f, 1.f, 1.f, 0.f };

		omm::Debug::Stats stats = RunOmmBake(0.5f, subdivisionLevel, { 4, 4 }, sizeof(triangleIndices), triangleIndices, texCoords, sizeof(texCoords), [](int i, int j)->float {

			uint32_t x = (i) % 2;
			uint32_t y = (j) % 2;

			float values[4] =
			{
				0.9f, 0.1f,
				0.1f, 0.7f
			};

			return 1.f - values[x + 2 * y];

			}, omm::Format::OC1_4_State);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
			.totalOpaque = 5132,
			.totalTransparent = 2394,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 666,
							});
		}
		else
		{
			ExpectEqual(stats, {
			.totalOpaque = 5132,
			.totalTransparent = 2393,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 667,
			});
		}
	}

	std::string CustomParamName(const ::testing::TestParamInfo<TestSuiteConfig>& info) {

		std::string str = "";
		if ((info.param & TestSuiteConfig::ComputeOnly) == TestSuiteConfig::ComputeOnly)
			str += "ComputeOnly_";
		if ((info.param & TestSuiteConfig::DisableSpecialIndices) == TestSuiteConfig::DisableSpecialIndices)
			str += "DisableSpecialIndices_";
		if ((info.param & TestSuiteConfig::Force32BitIndices) == TestSuiteConfig::Force32BitIndices)
			str += "Force32BitIndices_";
		if ((info.param & TestSuiteConfig::DisableTexCoordDeduplication) == TestSuiteConfig::DisableTexCoordDeduplication)
			str += "DisableTexCoordDeduplication_";
		if ((info.param & TestSuiteConfig::RedChannel) == TestSuiteConfig::RedChannel)
			str += "RedChannel_";
		if ((info.param & TestSuiteConfig::GreenChannel) == TestSuiteConfig::GreenChannel)
			str += "GreenChannel_";
		if ((info.param & TestSuiteConfig::BlueChannel) == TestSuiteConfig::BlueChannel)
			str += "BlueChannel_";
		if ((info.param & TestSuiteConfig::SetupBeforeBuild) == TestSuiteConfig::SetupBeforeBuild)
			str += "SetupBeforeBuild_";
		if ((info.param & TestSuiteConfig::EnablePostDispatchInfoStats) == TestSuiteConfig::EnablePostDispatchInfoStats)
			str += "PostDispatchInfoStats_";
		if (str.length() > 0)
			str.pop_back();

		if (str == "")
			str = "Default";

		return str;
	}

	INSTANTIATE_TEST_SUITE_P(OMMTestGPU, OMMBakeTestGPU, 
		::testing::Values(	
							    TestSuiteConfig::None,
							    TestSuiteConfig::EnablePostDispatchInfoStats,
							    TestSuiteConfig::DisableSpecialIndices,
							    TestSuiteConfig::DisableSpecialIndices | TestSuiteConfig::EnablePostDispatchInfoStats,
							    TestSuiteConfig::Force32BitIndices,
							    TestSuiteConfig::DisableTexCoordDeduplication,
							    TestSuiteConfig::RedChannel,
							    TestSuiteConfig::BlueChannel,
							    TestSuiteConfig::GreenChannel,
							    
							    TestSuiteConfig::SetupBeforeBuild,
								TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::EnablePostDispatchInfoStats,
							    TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::DisableSpecialIndices,
							    TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::DisableSpecialIndices | TestSuiteConfig::EnablePostDispatchInfoStats,
							    TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::Force32BitIndices,
							    TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::DisableTexCoordDeduplication,
							    TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::RedChannel,
							    TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::BlueChannel,
							    TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::GreenChannel,
								
							    TestSuiteConfig::ComputeOnly,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::EnablePostDispatchInfoStats,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::DisableSpecialIndices, 
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::DisableSpecialIndices | TestSuiteConfig::EnablePostDispatchInfoStats,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::Force32BitIndices,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::DisableTexCoordDeduplication,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::RedChannel,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::BlueChannel,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::GreenChannel,
							    
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::SetupBeforeBuild,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::EnablePostDispatchInfoStats,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::DisableSpecialIndices,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::DisableSpecialIndices | TestSuiteConfig::EnablePostDispatchInfoStats,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::Force32BitIndices,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::DisableTexCoordDeduplication,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::RedChannel,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::BlueChannel,
							    TestSuiteConfig::ComputeOnly | TestSuiteConfig::SetupBeforeBuild | TestSuiteConfig::GreenChannel
						), CustomParamName);

}  // namespace
