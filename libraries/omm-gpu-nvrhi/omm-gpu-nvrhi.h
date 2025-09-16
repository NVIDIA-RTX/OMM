/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <nvrhi/nvrhi.h>

#include <algorithm>
#include <functional>
#include <vector>
#include <memory>
#include <optional>

#include <omm.hpp>

namespace omm
{
	class GpuBakeNvrhiImpl;

	class GpuBakeNvrhi
	{
	public:

		// (Optional) In case the shaders are compiled externally the ShaderProvider can be provided 
		struct ShaderProvider
		{
			nvrhi::VulkanBindingOffsets bindingOffsets;
			std::function<nvrhi::ShaderHandle(nvrhi::ShaderType type, const char* shaderName, const char* shaderEntryName)> shaders;
		};

		using MessageCallback = std::function<void(omm::MessageSeverity severity, const char* message)>;

		enum class Operation
		{
			Invalid			= 0,
			Setup			= 1u << 0,
			Bake			= 1u << 1,
			SetupAndBake	= Setup | Bake
		};

		struct Input
		{
			Operation							operation = Operation::Invalid;
			nvrhi::TextureHandle				alphaTexture;
			uint32_t							alphaTextureChannel = 3;
			float								alphaCutoff = 0.5f;
			union {
				OMM_DEPRECATED_MSG("alphaCutoffGT is deprectated, please use alphaCutoffGreater")
				omm::OpacityState					alphaCutoffGT;
				omm::OpacityState					alphaCutoffGreater = omm::OpacityState::Opaque;
			};
			union {
				OMM_DEPRECATED_MSG("alphaCutoffLT is deprectated, please use alphaCutoffLessEqual")
				omm::OpacityState					alphaCutoffLT;
				omm::OpacityState					alphaCutoffLessEqual = omm::OpacityState::Transparent;
			};
			bool								bilinearFilter = true;
			bool								enableLevelLineIntersection = true;
			nvrhi::SamplerAddressMode			sampleMode = nvrhi::SamplerAddressMode::Clamp;

			nvrhi::Format						texCoordFormat = nvrhi::Format::R32_FLOAT;
			nvrhi::BufferHandle					texCoordBuffer;
			uint32_t							texCoordBufferOffsetInBytes = 0;
			uint32_t							texCoordStrideInBytes = 0;
			nvrhi::BufferHandle					indexBuffer;
			uint32_t							indexOffset = 0;
			uint32_t							indexBufferOffsetInBytes = 0; // _byte_ offset when creating the input descriptor (indexOffset is preferred, this is kept for backwards comp.)
			uint32_t							numIndices = 0;

			uint32_t							maxSubdivisionLevel = 0;
			uint32_t							maxOutOmmArraySize = 0xFFFFFFFF;
			nvrhi::rt::OpacityMicromapFormat	format = nvrhi::rt::OpacityMicromapFormat::OC1_4_State;
			float								dynamicSubdivisionScale = 0.5f;
			bool								minimalMemoryMode = false;
			bool								enableStats = false;
			bool								enableSpecialIndices = true;
			bool								force32BitIndices = false;
			bool								enableTexCoordDeduplication = true;
			bool								computeOnly = false;
			bool								enableNsightDebugMode = false;
		};

		struct PreDispatchInfo
		{
			nvrhi::Format	ommIndexFormat;
			uint32_t		ommIndexCount;
			size_t			ommIndexBufferSize;
			size_t			ommIndexHistogramSize;
			size_t			ommArrayBufferSize;
			size_t			ommDescBufferSize;
			size_t			ommDescArrayHistogramSize;
			size_t			ommPostDispatchInfoBufferSize;
		};

		struct Buffers
		{
			nvrhi::BufferHandle ommArrayBuffer;
			nvrhi::BufferHandle ommDescBuffer;
			nvrhi::BufferHandle ommIndexBuffer;
			nvrhi::BufferHandle ommDescArrayHistogramBuffer;
			nvrhi::BufferHandle ommIndexHistogramBuffer;
			nvrhi::BufferHandle ommPostDispatchInfoBuffer;

			uint32_t ommArrayBufferOffset = 0;
			uint32_t ommDescBufferOffset = 0;
			uint32_t ommIndexBufferOffset = 0;
			uint32_t ommDescArrayHistogramBufferOffset = 0;
			uint32_t ommIndexHistogramBufferOffset = 0;
			uint32_t ommPostDispatchInfoBufferOffset = 0;
		};

		struct PostDispatchInfo
		{
			uint32_t ommArrayBufferSize;
			uint32_t ommDescBufferSize;
			uint32_t ommTotalOpaqueCount;
			uint32_t ommTotalTransparentCount;
			uint32_t ommTotalUnknownCount;
			uint32_t ommTotalFullyOpaqueCount;
			uint32_t ommTotalFullyTransparentCount;
			uint32_t ommTotalFullyUnknownCount;
		};

		struct Stats
		{
			uint64_t totalOpaque = 0;
			uint64_t totalTransparent = 0;
			uint64_t totalUnknownTransparent = 0;
			uint64_t totalUnknownOpaque = 0;
			uint32_t totalFullyOpaque = 0;
			uint32_t totalFullyTransparent = 0;
			uint32_t totalFullyUnknownOpaque = 0;
			uint32_t totalFullyUnknownTransparent = 0;
		};

		GpuBakeNvrhi(nvrhi::DeviceHandle device, nvrhi::CommandListHandle commandList, bool enableDebug, ShaderProvider* shaderProvider = nullptr, std::optional<MessageCallback> callback = nullptr);
		~GpuBakeNvrhi();

		// CPU side pre-build info.
		void GetPreDispatchInfo(const Input& params, PreDispatchInfo& info);

		void Dispatch(
			nvrhi::CommandListHandle commandList,
			const Input& params,
			const Buffers& buffers);

		void Clear();

		static void ReadPostDispatchInfo(void* pData, size_t byteSize, PostDispatchInfo& outPostDispatchInfo);
		static void ReadUsageDescBuffer(void* pData, size_t byteSize, std::vector<nvrhi::rt::OpacityMicromapUsageCount>& outVmUsages);

		// Debug dumping
		void DumpDebug(
			const char* folderName,
			const char* debugName,
			const Input& params,
			const std::vector<uint8_t>& ommArrayBuffer,
			const std::vector<uint8_t>& ommDescBuffer,
			const std::vector<uint8_t>& ommIndexBuffer,
			nvrhi::Format ommIndexBufferFormat,
			const std::vector<uint8_t>& ommDescArrayHistogramBuffer,
			const std::vector<uint8_t>& ommIndexHistogramBuffer,
			const void* indexBuffer,
			const uint32_t indexCount,
			nvrhi::Format ommTexCoordBufferFormat,
			const void* texCoords,
			const float* imageData,
			const uint32_t width,
			const uint32_t height
		);

		Stats GetStats(const omm::Cpu::BakeResultDesc& desc);

	private:
		std::unique_ptr< GpuBakeNvrhiImpl> m_impl;
	};
} // namespace omm
