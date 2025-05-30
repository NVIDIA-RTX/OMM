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

#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <donut/core/vfs/VFS.h>
#include <donut/render/MipMapGenPass.h>
#include <donut/render/PixelReadbackPass.h>
#include <nvrhi/utils.h>
#include <donut/app/imgui_console.h>
#include <donut/app/imgui_renderer.h>
#include <imfilebrowser.h>
#include "OpenSans_Regular.h"

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/background_vs_ps_main_ps.dxbc.h"
#include "compiled_shaders/background_vs_ps_main_vs.dxbc.h"
#include "compiled_shaders/shaders_main_ps.dxbc.h"
#include "compiled_shaders/shaders_main_vs.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/background_vs_ps_main_ps.dxil.h"
#include "compiled_shaders/background_vs_ps_main_vs.dxil.h"
#include "compiled_shaders/shaders_main_ps.dxil.h"
#include "compiled_shaders/shaders_main_vs.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/background_vs_ps_main_ps.spirv.h"
#include "compiled_shaders/background_vs_ps_main_vs.spirv.h"
#include "compiled_shaders/shaders_main_ps.spirv.h"
#include "compiled_shaders/shaders_main_vs.spirv.h"
#endif
#endif

using namespace donut;
using namespace donut::math;

#include "shader_cb.h"

#include <fstream>
#include <vector>
#include <cstdint>

#include <omm.hpp>

using namespace donut;

static const char* g_WindowTitle = "Opacity Micro-Map Viewer Tool";

static const char* ToString(omm::Result result)
{
    switch (result)
    {
    case omm::Result::SUCCESS: return "SUCCESS";
    case omm::Result::FAILURE: return "FAILURE";
    case omm::Result::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case omm::Result::INSUFFICIENT_SCRATCH_MEMORY: return "INSUFFICIENT_SCRATCH_MEMORY";
    case omm::Result::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
    case omm::Result::WORKLOAD_TOO_BIG: return "WORKLOAD_TOO_BIG";
    case omm::Result::MAX_NUM: return "MAX_NUM";
    default:
        return "unknown error code";
    }
}

static void AbortOnFailure(const char* funName, omm::Result result)
{
    donut::log::fatal("%s returned %s", funName, ToString(result));
}

static void PopupOnFailure(const char* funName, omm::Result result)
{
    donut::log::error("%s returned %s", funName, ToString(result));
}

static void Log(omm::MessageSeverity severity, const char* message, void* userArg)
{
    donut::log::Severity donutSeverity = donut::log::Severity::Info;
    switch (severity)
    {
    case omm::MessageSeverity::Info:
        donutSeverity = donut::log::Severity::Info;
        break;
    case omm::MessageSeverity::Warning:
        donutSeverity = donut::log::Severity::Warning;
        break;
    case omm::MessageSeverity::PerfWarning:
        donutSeverity = donut::log::Severity::Warning;
        break;
    case omm::MessageSeverity::Fatal:
        donutSeverity = donut::log::Severity::Error;
        break;
    }

    donut::log::message(donutSeverity, "[omm-sdk]: %s", message);
}

#define _OMM_ON_ERROR(fun, onError)do { \
    omm::Result res_##__LINE__ = fun; \
    if (res_##__LINE__ != omm::Result::SUCCESS) \
    { \
        onError(#fun, res_##__LINE__); \
    } \
    } while (false)

#define OMM_ABORT_ON_ERROR(fun) _OMM_ON_ERROR(fun, AbortOnFailure)

#define OMM_POPUP_ON_ERROR(fun) _OMM_ON_ERROR(fun, PopupOnFailure)

class OmmLibrary
{
public:
    OmmLibrary():_baker(0)
    {
        omm::BakerCreationDesc desc;
        desc.type = omm::BakerType::CPU;
        desc.messageInterface.messageCallback = &Log;

        OMM_ABORT_ON_ERROR(omm::CreateBaker(desc, &_baker));
    }

    ~OmmLibrary()
    {
        if (_baker != 0)
        {
            OMM_ABORT_ON_ERROR(omm::DestroyBaker(_baker));
        }
    }

    omm::Baker GetBaker() const { return _baker; }

private:
    omm::Baker _baker;
};

struct UIData
{
    bool ShowUI = true;

    int primitiveStart = 0;
    int primitiveEnd = -1;
    int ommIndexHighlight = -5;
    int ommIndexIsolate = -1;
    float zoom = 1.f;
    float2 offset = 0.f;
    float2 prevOffset = 0.f;
    float alphaVal = 0.f;
    int2 texel = int2(0, 0);

    std::string path;
    std::vector<std::filesystem::path> ommFiles;
    int selectedFile = 0;

    std::optional<omm::Cpu::TextureDesc> textureDesc;
    std::optional<omm::Cpu::BakeInputDesc> input;

    bool load = false;
    bool rebake = false;
    bool recompile = true;

    // viz
    bool drawAlphaContour = true;
    bool drawWireFrame = true;
    bool drawMicroTriangles = true;
    bool colorizeStates = true;
    bool ommIndexHighlightEnable = true;
};

class OmmGpuData
{
    OmmLibrary _lib;
    UIData& m_ui;
    nvrhi::IDevice* m_device = nullptr;
    nvrhi::CommandListHandle m_commandList = nullptr;
    std::shared_ptr<engine::ShaderFactory> m_shaderFactory;

    bool _hasLoadedData = false;
    std::string _fileName;
    std::vector<uint8_t> _data;

    nvrhi::SamplerHandle _samplerLinear;
    nvrhi::SamplerHandle _samplerPoint;
    nvrhi::TextureHandle _alphaTexture = nullptr;
    nvrhi::TextureHandle _alphaTextureMin = nullptr;
    nvrhi::TextureHandle _alphaTextureMax = nullptr;
    nvrhi::BufferHandle _texCoordBuffer = nullptr;
    nvrhi::BufferHandle _indexBuffer = nullptr;
    nvrhi::BufferHandle _ommIndexBuffer = nullptr;
    nvrhi::BufferHandle _ommDesc = nullptr;
    nvrhi::BufferHandle _ommArrayData = nullptr;

    omm::Cpu::TextureDesc _textureDesc;
    omm::Cpu::BakeInputDesc _input;
    omm::Cpu::BakeResult _result = 0;
    const omm::Cpu::BakeResultDesc* _resultDesc = nullptr;
    omm::Debug::Stats _stats;

    uint64_t _bakeTimeInMs = 0;
    uint64_t _bakeTimeInSeconds = 0;
    uint32_t _indexCount = 0;
    uint32_t _ommIndexCount = 0;

public:
    OmmGpuData(UIData& ui):m_ui(ui)
    {
    }

    void Init(nvrhi::IDevice* device, std::shared_ptr<engine::ShaderFactory> shaderFactory)
    {
        m_device = device;
        m_commandList = device->createCommandList();
        m_shaderFactory = shaderFactory;
    }

    ~OmmGpuData()
    {
        ClearAll();
    }

    bool HasLoadedData() const
    {
        return _hasLoadedData;
    }

    bool Load(const std::string& fileName)
    {
        ClearAll();
        _LoadOmmData(fileName);

        return _RebuildOmmData(true /*loadOnly*/);
    }

    bool Bake()
    {
        ClearAll();
        return _RebuildOmmData(false /*loadOnly*/);
    }

    void ClearAll()
    {
        _hasLoadedData = false;
        if (_result != 0)
        {
            OMM_ABORT_ON_ERROR(omm::Cpu::DestroyBakeResult(_result));
            _result = 0;
        }
        _resultDesc = nullptr;

        _stats = {};

        // _sampler = nullptr;
        _alphaTexture = nullptr;
        _texCoordBuffer = nullptr;
        _indexBuffer = nullptr;
        _ommIndexBuffer = nullptr;
        _ommDesc = nullptr;
        _ommArrayData = nullptr;
    }

    std::string GetFileName() const { return _fileName; }

    nvrhi::SamplerHandle GetSamplerLinear() const { return _samplerLinear; }
    nvrhi::SamplerHandle GetSamplerPoint() const { return _samplerPoint; }
    nvrhi::TextureHandle GetAlphaTexture() const { return _alphaTexture; }
    nvrhi::TextureHandle GetAlphaTextureMin() const { return _alphaTextureMin; }
    nvrhi::TextureHandle GetAlphaTextureMax() const { return _alphaTextureMax; }
    nvrhi::BufferHandle GetIndexBuffer() const { return _indexBuffer; }
    nvrhi::BufferHandle GetTexCoordBuffer() const { return _texCoordBuffer; }
    nvrhi::BufferHandle GetOmmIndexBuffer() const { return _ommIndexBuffer; }
    nvrhi::BufferHandle GetOmmDesc() const { return _ommDesc; }
    nvrhi::BufferHandle GetOmmArrayData() const { return _ommArrayData; }

    uint32_t GetIndexCount() const { return _indexCount; }
    uint32_t GetOmmIndexCount() const { return _ommIndexCount; }
    const omm::Cpu::TextureDesc& GetDefaultTextureDesc() const { return _textureDesc; }
    const omm::Cpu::BakeInputDesc& GetDefaultInput() const { return _input; }
    const omm::Cpu::BakeResultDesc* GetResult() const { return _resultDesc; }
    const omm::Debug::Stats& GetStats() const { return _stats; }
    const uint64_t GetBakeTimeInMs() const { return _bakeTimeInMs; }
    const uint64_t GetBakeTimeInSeconds() const { return _bakeTimeInSeconds; }

private:
    std::vector<uint8_t> _LoadDataFile(const std::string& fileName)
    {
        _fileName = fileName;
        // Open the file in binary mode and position the file pointer at the end
        std::ifstream file(fileName, std::ios::binary | std::ios::ate);

        if (!file) {
            throw std::runtime_error("Failed to open file: " + fileName);
        }

        // Determine the file size
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);  // Go back to the beginning of the file

        // Create a vector with enough space and read the file contents into it
        std::vector<uint8_t> buffer(size);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read file: " + fileName);
        }
        return buffer;
    }

    static nvrhi::SamplerAddressMode GetSampler(omm::TextureAddressMode addressMode)
    {
        switch (addressMode)
        {
        case omm::TextureAddressMode::Wrap: return nvrhi::SamplerAddressMode::Wrap;
        case omm::TextureAddressMode::Mirror: return nvrhi::SamplerAddressMode::Mirror;
        case omm::TextureAddressMode::Clamp: return nvrhi::SamplerAddressMode::Clamp;
        case omm::TextureAddressMode::Border: return nvrhi::SamplerAddressMode::Border;
        case omm::TextureAddressMode::MirrorOnce: return nvrhi::SamplerAddressMode::MirrorOnce;
        default:
        {
            assert(false);
            return nvrhi::SamplerAddressMode::Wrap;
        }
        }
    }

    void _InitSampler(const omm::Cpu::BakeInputDesc& input)
    {
        nvrhi::SamplerAddressMode addressMode = GetSampler(input.runtimeSamplerDesc.addressingMode);
        
        auto samplerDesc = nvrhi::SamplerDesc()
            .setAllAddressModes(addressMode);
        samplerDesc.setAllFilters(true);
        _samplerLinear = m_device->createSampler(samplerDesc);
        samplerDesc.setAllFilters(false);
        _samplerPoint = m_device->createSampler(samplerDesc);
    }

    void _InitTexture(const omm::Cpu::TextureDesc& ommTex)
    {
        nvrhi::TextureDesc d;
        d.height = ommTex.mips[0].height;
        d.width = ommTex.mips[0].width;
        d.mipLevels = (uint)( std::log2f((float)std::max(ommTex.mips[0].height, ommTex.mips[0].width)) + 0.5f);

        d.format = ommTex.format == omm::Cpu::TextureFormat::FP32 ? nvrhi::Format::R32_FLOAT : nvrhi::Format::R8_UNORM;
        d.initialState = nvrhi::ResourceStates::ShaderResource;
        d.keepInitialState = true;
        d.isUAV = true;
        d.debugName = "AlphaTexture";
        _alphaTexture = m_device->createTexture(d);
        _alphaTextureMin = m_device->createTexture(d);
        _alphaTextureMax = m_device->createTexture(d);

        size_t texelSize = ommTex.format == omm::Cpu::TextureFormat::FP32 ? sizeof(float) : sizeof(uint8_t);

        m_commandList->open();
        m_commandList->setEnableAutomaticBarriers(true);
        m_commandList->writeTexture(_alphaTexture, 0, 0, ommTex.mips[0].textureData, texelSize * ommTex.mips[0].rowPitch);
        m_commandList->copyTexture(_alphaTextureMin, nvrhi::TextureSlice().setMipLevel(0), _alphaTexture, nvrhi::TextureSlice().setMipLevel(0));
        m_commandList->copyTexture(_alphaTextureMax, nvrhi::TextureSlice().setMipLevel(0), _alphaTexture, nvrhi::TextureSlice().setMipLevel(0));

        donut::render::MipMapGenPass mipMapAvg(m_device, m_shaderFactory, _alphaTexture, donut::render::MipMapGenPass::MODE_COLOR);
        mipMapAvg.Dispatch(m_commandList);

        donut::render::MipMapGenPass mipMapMin(m_device, m_shaderFactory, _alphaTextureMin, donut::render::MipMapGenPass::MODE_MIN);
        mipMapMin.Dispatch(m_commandList);

        donut::render::MipMapGenPass mipMapMax(m_device, m_shaderFactory, _alphaTextureMax, donut::render::MipMapGenPass::MODE_MAX);
        mipMapMax.Dispatch(m_commandList);

        m_commandList->close();
        m_device->executeCommandList(m_commandList);
        m_device->waitForIdle();
    }

    void _InitBuffers(const omm::Cpu::BakeInputDesc& input)
    {
        if (input.indexCount != 0)
        {
            nvrhi::BufferDesc ib;
            ib.debugName = "IndexBuffer";
            size_t indexBufferFormatSize;
            nvrhi::Format indexBufferFormat;
            if (input.indexFormat == omm::IndexFormat::UINT_8)
            {
                indexBufferFormatSize = 1;
                indexBufferFormat = nvrhi::Format::R8_UINT;
            }
            else if (input.indexFormat == omm::IndexFormat::UINT_16)
            {
                indexBufferFormatSize = 2;
                indexBufferFormat = nvrhi::Format::R16_UINT;
            }
            else // omm::IndexFormat::UINT_32
            {
                indexBufferFormatSize = 4;
                indexBufferFormat = nvrhi::Format::R32_UINT;
            }      
            ib.byteSize = input.indexCount * indexBufferFormatSize;
            ib.format = indexBufferFormat;
            ib.initialState = nvrhi::ResourceStates::ShaderResource;
            ib.keepInitialState = true;
            ib.isIndexBuffer = true;
            _indexBuffer = m_device->createBuffer(ib);
        }
        else
        {
            _indexBuffer = nullptr;
        }

        {
            _indexCount = input.indexCount;

            uint32_t maxTexCoordIndex = 0;
            for (uint32_t i = 0; i < input.indexCount; ++i)
            {
                if (input.indexFormat == omm::IndexFormat::UINT_8)
                {
                    uint8_t val = ((uint8_t*)input.indexBuffer)[i];
                    maxTexCoordIndex = std::max<uint32_t>(maxTexCoordIndex, val);
                }
                else if (input.indexFormat == omm::IndexFormat::UINT_16)
                {
                    uint16_t val = ((uint16_t*)input.indexBuffer)[i];
                    maxTexCoordIndex = std::max<uint32_t>(maxTexCoordIndex, val);
                }
                else
                {
                    assert(input.indexFormat == omm::IndexFormat::UINT_32);
                    uint32_t val = ((uint32_t*)input.indexBuffer)[i];
                    maxTexCoordIndex = std::max(maxTexCoordIndex, val);
                }
            }

            const size_t texCoordBufferSize = (maxTexCoordIndex + 1) * (input.texCoordFormat == omm::TexCoordFormat::UV32_FLOAT ? 4 : 2);

            if (texCoordBufferSize != 0)
            {
                assert(input.texCoordFormat == omm::TexCoordFormat::UV32_FLOAT);

                nvrhi::BufferDesc texCoord;
                texCoord.debugName = "TexCoordBuffer";
                texCoord.byteSize = texCoordBufferSize * 2;
                texCoord.format = input.texCoordFormat == omm::TexCoordFormat::UV32_FLOAT ? nvrhi::Format::RG32_FLOAT : nvrhi::Format::RG16_FLOAT;
                texCoord.initialState = nvrhi::ResourceStates::ShaderResource;
                texCoord.keepInitialState = true;
                texCoord.isVertexBuffer = true;
                _texCoordBuffer = m_device->createBuffer(texCoord);
            }
            else
            {
                _texCoordBuffer = nullptr;
            }
        }

        m_commandList->open();
        m_commandList->setEnableAutomaticBarriers(true);
        if (_indexBuffer)
            m_commandList->writeBuffer(_indexBuffer, input.indexBuffer, _indexBuffer->getDesc().byteSize);
        if (_texCoordBuffer)
            m_commandList->writeBuffer(_texCoordBuffer, input.texCoords, _texCoordBuffer->getDesc().byteSize);
        m_commandList->close();
        m_device->executeCommandList(m_commandList);
        m_device->waitForIdle();
    }

    void _InitBakeResults(const omm::Cpu::BakeInputDesc& input)
    {
        {
            nvrhi::BufferDesc ommIB;
            ommIB.debugName = "OmmIndexBuffer";
            if (_resultDesc && _resultDesc->indexCount != 0)
            {
                if (_resultDesc->indexFormat == omm::IndexFormat::UINT_8)
                {
                    ommIB.format = nvrhi::Format::R8_SINT;
                    ommIB.byteSize = _resultDesc->indexCount * 1;
                }
                else if (_resultDesc->indexFormat == omm::IndexFormat::UINT_16)
                {
                    ommIB.format = nvrhi::Format::R16_SINT;
                    ommIB.byteSize = _resultDesc->indexCount * 2;
                }
                else // omm::IndexFormat::UINT_32
                {
                    ommIB.format = nvrhi::Format::R32_SINT;
                    ommIB.byteSize = _resultDesc->indexCount * 4;
                }
            }
            else
            {
                ommIB.format = nvrhi::Format::R32_SINT;
                ommIB.byteSize = 8;
            }

            ommIB.initialState = nvrhi::ResourceStates::ShaderResource;
            ommIB.keepInitialState = true;
            ommIB.canHaveTypedViews = true;
            _ommIndexBuffer = m_device->createBuffer(ommIB);
        }

        {
            _ommIndexCount = _resultDesc ? _resultDesc->descArrayCount : 0;

            nvrhi::BufferDesc ommDesc;
            ommDesc.debugName = "OmmDescBuffer";
            if (_resultDesc && _resultDesc->descArrayCount != 0)
                ommDesc.byteSize = _resultDesc->descArrayCount * sizeof(omm::Cpu::OpacityMicromapDesc);
            else
                ommDesc.byteSize = 8;
            ommDesc.format = nvrhi::Format::UNKNOWN;
            ommDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            ommDesc.structStride = sizeof(omm::Cpu::OpacityMicromapDesc);
            ommDesc.keepInitialState = true;
            _ommDesc = m_device->createBuffer(ommDesc);
        }

        {
            nvrhi::BufferDesc ommArray;
            ommArray.debugName = "OmmArrayBuffer";
            if (_resultDesc && _resultDesc->arrayDataSize != 0)
                ommArray.byteSize = _resultDesc->arrayDataSize;
            else
                ommArray.byteSize = 8;
            //ommArray.format = nvrhi::Format::R32_UINT;
            ommArray.initialState = nvrhi::ResourceStates::ShaderResource;
            ommArray.keepInitialState = true;
            ommArray.canHaveRawViews = true;
            _ommArrayData = m_device->createBuffer(ommArray);
        }

        m_commandList->open();
        m_commandList->setEnableAutomaticBarriers(true);
        if (_resultDesc && _resultDesc->indexCount != 0)
            m_commandList->writeBuffer(_ommIndexBuffer, _resultDesc->indexBuffer, _ommIndexBuffer->getDesc().byteSize);
        if (_resultDesc && _resultDesc->descArrayCount != 0)
            m_commandList->writeBuffer(_ommDesc, _resultDesc->descArray, _ommDesc->getDesc().byteSize);
        if (_resultDesc && _resultDesc->arrayDataSize != 0)
            m_commandList->writeBuffer(_ommArrayData, _resultDesc->arrayData, _ommArrayData->getDesc().byteSize);
        m_commandList->close();
        m_device->executeCommandList(m_commandList);
        m_device->waitForIdle();
    }

    void _LoadOmmData(const std::string& fileName)
    {
        if (_fileName != fileName)
        {
            m_ui.primitiveEnd = -1;

            m_ui.input.reset();
            m_ui.textureDesc.reset();
            _data = _LoadDataFile(fileName);
        }
    }

    bool _RebuildOmmData(bool loadOnly)
    {
        omm::Cpu::BlobDesc blobDesc;
        blobDesc.data = _data.data();
        blobDesc.size = _data.size();

        omm::Baker baker = _lib.GetBaker();

        omm::Cpu::DeserializedResult res;
        {
            omm::Result err = omm::Cpu::Deserialize(baker, blobDesc, &res);
            OMM_POPUP_ON_ERROR(err);
            if (err != omm::Result::SUCCESS)
                return false;
        }

        const omm::Cpu::DeserializedDesc* deserializeDesc = nullptr;
        OMM_ABORT_ON_ERROR(omm::Cpu::GetDeserializedDesc(res, &deserializeDesc));

        assert(deserializeDesc->numInputDescs > 0);

        nvrhi::TextureHandle tex;
        
        omm::Cpu::BakeInputDesc input = deserializeDesc->inputDescs[0];
        _input = input;

        if (!m_ui.input.has_value())
        {
            m_ui.input = input;
            reinterpret_cast<uint32_t&>(m_ui.input->bakeFlags) |= uint32_t(omm::Cpu::BakeFlags::EnableInternalThreads);
            m_ui.input->maxWorkloadSize = 0xFFFFFFFFFFFFFFFF;
        }

        input = m_ui.input.value();
            
        input.texture = _input.texture;

        input.texCoords = _input.texCoords;
        input.texCoordStrideInBytes = _input.texCoordStrideInBytes;
        input.texCoordFormat = _input.texCoordFormat;
            
        input.indexFormat = _input.indexFormat;
        input.indexBuffer = _input.indexBuffer;
        input.indexCount = _input.indexCount;

        input.subdivisionLevels = _input.subdivisionLevels;
        
        omm::Cpu::TextureMipDesc mips[16];
        omm::Cpu::TextureDesc texDesc;
        texDesc.mips = mips;
        texDesc.mipCount = 1; // TODO

        OMM_ABORT_ON_ERROR(omm::Cpu::GetTextureDesc(input.texture, &texDesc));
        const size_t size = texDesc.format == omm::Cpu::TextureFormat::FP32 ? sizeof(float) : sizeof(uint8_t);

        // Todo: figure out the conservative memory bound...
        size_t maxDim = std::max(std::max(texDesc.mips[0].rowPitch, texDesc.mips[0].height), texDesc.mips[0].width);
        std::vector<uint8_t> textureData(size * maxDim * maxDim);

        mips[0].textureData = (const void*)textureData.data();

        OMM_ABORT_ON_ERROR(omm::Cpu::GetTextureDesc(input.texture, &texDesc));

        if (!m_ui.textureDesc.has_value())
        {
            m_ui.textureDesc = texDesc;
        }

        _InitBuffers(input);
        _InitSampler(input);
        _InitTexture(texDesc);

        if (loadOnly)
        {
            _InitBakeResults(input);
            OMM_ABORT_ON_ERROR(omm::Cpu::DestroyDeserializedResult(res));
            _hasLoadedData = true;
            return true;
        }

        {
            texDesc = m_ui.textureDesc.value();
            texDesc.mips = mips;

            omm::Cpu::Texture textureClone;
            OMM_ABORT_ON_ERROR(omm::Cpu::CreateTexture(baker, texDesc, &textureClone));

            input.texture = textureClone;

            auto start = std::chrono::high_resolution_clock::now();
            omm::Result res = omm::Cpu::Bake(baker, input, &_result);
            OMM_POPUP_ON_ERROR(res);
            auto end = std::chrono::high_resolution_clock::now();
            _bakeTimeInMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            _bakeTimeInSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

            if (res == omm::Result::SUCCESS)
            {
                OMM_ABORT_ON_ERROR(omm::Cpu::GetBakeResultDesc(_result, &_resultDesc));

                OMM_ABORT_ON_ERROR(omm::Debug::GetStats2(baker, _result, &_stats));
            }

            OMM_ABORT_ON_ERROR(omm::Cpu::DestroyTexture(baker, textureClone));
        }

        _InitBakeResults(input);

        OMM_ABORT_ON_ERROR(omm::Cpu::DestroyDeserializedResult(res));

        _hasLoadedData = true;
        return true;
    }
};

class BasicTriangle : public app::IRenderPass
{
private:
    nvrhi::BufferHandle m_ConstantBuffer;
    nvrhi::TextureHandle m_ReadbackTexture;
    nvrhi::ShaderHandle m_VertexShader;
    nvrhi::ShaderHandle m_PixelShader;

    std::shared_ptr<donut::render::PixelReadbackPass> m_pixelReadback;

    nvrhi::ShaderHandle m_BackgroundVS;
    nvrhi::ShaderHandle m_BackgroundPS;
    nvrhi::GraphicsPipelineHandle m_BackgroundPSO;

    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingSetHandle m_BindingSets;
    nvrhi::GraphicsPipelineHandle m_Pipeline;
    nvrhi::GraphicsPipelineHandle m_PipelineWireFrame;

    nvrhi::SamplerHandle m_LinearSampler;
    nvrhi::InputLayoutHandle m_InputLayout;
    nvrhi::CommandListHandle m_CommandList;
    OmmGpuData m_ommData;
    UIData& m_ui;
    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;

    bool m_mouseDown = false;
    float2 m_mousePos = float2(0, 0);
    float2 m_referencePos = float2(0, 0);
public:
    using IRenderPass::IRenderPass;

    BasicTriangle(app::DeviceManager* deviceManager, UIData& ui):IRenderPass(deviceManager),m_ui(ui), m_ommData(ui) {}

    std::shared_ptr<engine::ShaderFactory> GetShaderFactory()
    {
        return m_ShaderFactory;
    }

    const OmmGpuData& GetOmmGpuData() const
    {
        return m_ommData;
    }

    bool Init()
    {
        auto rootFs = std::make_shared<vfs::RootFileSystem>();

        m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFs, "");

        m_CommandList = GetDevice()->createCommandList();

        m_ommData.Init(GetDevice(), m_ShaderFactory);
        //m_ommData.Load(m_ommFiles[0].string());
        //m_ommData.Bake();
    
        return true;
    }

protected:

    virtual bool KeyboardUpdate(int key, int scancode, int action, int mods)
    {
        if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
        {
            auto& files = m_ui.ommFiles;
            m_ui.selectedFile = (m_ui.selectedFile + 1) % m_ui.ommFiles.size();
            m_ui.load = true;
            m_ui.rebake = true;
        } 
        else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
        {
            auto& files = m_ui.ommFiles;
            m_ui.selectedFile = uint(m_ui.selectedFile - 1) % m_ui.ommFiles.size();
            m_ui.load = true;
            m_ui.rebake = true;
        }
        else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        {
            m_ui.rebake = true;
        }
        return false;
    }

    virtual bool MouseButtonUpdate(int button, int action, int mods)
    {
        if (button == 0)
        {
            if (action == 1)
            {
                m_mouseDown = true;
                m_referencePos = m_mousePos;
            }
            else
            {
                m_ui.prevOffset += m_ui.offset;
                m_ui.offset = float2(0, 0);
                m_mouseDown = false;
            }
        }
        return false;
    }

    virtual bool MouseScrollUpdate(double xoffset, double yoffset)
    {
        m_ui.zoom += 0.15f * m_ui.zoom * (float)yoffset;

        return false;
    }

    virtual bool MousePosUpdate(double xpos, double ypos)
    {
        float2 aspectRatioTex(1.f, 1.f);

        if (nvrhi::TextureHandle alphaTex = m_ommData.GetAlphaTexture())
        {
            aspectRatioTex = float2((float)alphaTex->getDesc().width / (float)alphaTex->getDesc().height, 1.f);
        }

        int2 windowSize;
        GetDeviceManager()->GetWindowDimensions(windowSize.x, windowSize.y);
        windowSize.x = windowSize.y;

        m_mousePos = float2((float)xpos, (float)ypos);
        
        if (m_mouseDown)
        {
            m_ui.offset = (2.f / ((float2)windowSize * aspectRatioTex)) * (m_referencePos - m_mousePos) / m_ui.zoom;
            m_ui.offset.x = -m_ui.offset.x;
        }
        return false;
    }

private:

    void ClearAllResource()
    {
        m_Pipeline = nullptr;
        m_BindingLayout = nullptr;
        m_BindingSets = nullptr;
        m_BackgroundPSO = nullptr;
        m_VertexShader = nullptr;
        m_PixelShader = nullptr;
        m_BackgroundVS = nullptr;
        m_BackgroundPS = nullptr;

        m_ShaderFactory->ClearCache();
    }

    void BackBufferResizing() override
    { 
        ClearAllResource();
        m_ui.recompile = false;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }
    
    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        if (m_ui.rebake || m_ui.load || m_ui.recompile)
        {
            GetDevice()->waitForIdle();
            if (m_ui.rebake || m_ui.load)
                m_ommData.ClearAll();

            if (m_ui.load && m_ui.selectedFile >= 0)
            {
                if (m_ommData.Load(m_ui.ommFiles[m_ui.selectedFile].string()))
                {
                    if (m_ui.rebake)
                    {
                        m_ommData.Bake();
                    }
                }
                m_ui.rebake = false;
                m_ui.load = false;
            }

            if (m_ui.rebake)
            {
                m_ommData.Bake();
                m_ui.rebake = false;
            }
            
            ClearAllResource();
            m_ui.load = false;
            m_ui.recompile = false;
        }

        if (!m_ommData.HasLoadedData())
        {
            m_CommandList->open();
            nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
            m_CommandList->close();
            GetDevice()->executeCommandList(m_CommandList);
            return;
        }

        if (!m_Pipeline)
        {
            nvrhi::TextureDesc d;
            d.height = framebuffer->getFramebufferInfo().height;
            d.width = framebuffer->getFramebufferInfo().width;
            d.format = nvrhi::Format::RGBA32_FLOAT;
            d.initialState = nvrhi::ResourceStates::ShaderResource;
            d.keepInitialState = true;
            d.isUAV = true;
            d.debugName = "ReadbackTexture";
            m_ReadbackTexture = GetDevice()->createTexture(d);

            m_pixelReadback = std::make_shared<donut::render::PixelReadbackPass>(GetDevice(), m_ShaderFactory, m_ReadbackTexture.Get(), nvrhi::Format::RGBA32_FLOAT);

            m_VertexShader = m_ShaderFactory->CreateStaticPlatformShader(DONUT_MAKE_PLATFORM_SHADER(g_shaders_main_vs), nullptr, nvrhi::ShaderType::Vertex);
            m_PixelShader = m_ShaderFactory->CreateStaticPlatformShader(DONUT_MAKE_PLATFORM_SHADER(g_shaders_main_ps), nullptr, nvrhi::ShaderType::Pixel);

            m_BackgroundVS = m_ShaderFactory->CreateStaticPlatformShader(DONUT_MAKE_PLATFORM_SHADER(g_background_vs_ps_main_vs), nullptr, nvrhi::ShaderType::Vertex);
            m_BackgroundPS = m_ShaderFactory->CreateStaticPlatformShader(DONUT_MAKE_PLATFORM_SHADER(g_background_vs_ps_main_ps), nullptr, nvrhi::ShaderType::Pixel);

            m_ConstantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(Constants), "Constants", 16));

            nvrhi::VertexAttributeDesc vertexAttr;
            vertexAttr.name = "SV_POSITION";
            vertexAttr.format = nvrhi::Format::RG32_FLOAT;
            vertexAttr.elementStride = sizeof(float) * 2;
            m_InputLayout = GetDevice()->createInputLayout(&vertexAttr, 1, m_VertexShader);

            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
                nvrhi::BindingSetItem::Sampler(0, m_ommData.GetSamplerLinear()),
                nvrhi::BindingSetItem::Sampler(1, m_ommData.GetSamplerPoint()),
                nvrhi::BindingSetItem::Texture_SRV(0, m_ommData.GetAlphaTexture()),
                nvrhi::BindingSetItem::Texture_SRV(1, m_ommData.GetAlphaTextureMin()),
                nvrhi::BindingSetItem::Texture_SRV(2, m_ommData.GetAlphaTextureMax()),
                nvrhi::BindingSetItem::TypedBuffer_SRV(3, m_ommData.GetOmmIndexBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_ommData.GetOmmDesc()),
                nvrhi::BindingSetItem::RawBuffer_SRV(5, m_ommData.GetOmmArrayData()),
                nvrhi::BindingSetItem::Texture_UAV(0, m_ReadbackTexture),
            };

            // Create the binding layout (if it's empty -- so, on the first iteration) and the binding set.
            if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_BindingLayout, m_BindingSets))
            {
                log::error("Couldn't create the binding set or layout");
            }


            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS = m_VertexShader;
            psoDesc.PS = m_PixelShader;
            psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
            psoDesc.renderState.depthStencilState.depthTestEnable = false;
           // psoDesc.renderState.depthStencilState.stencilEnable = false;
           // psoDesc.renderState.depthStencilState.frontFaceStencil.stencilFunc = nvrhi::ComparisonFunc::Equal;
           // psoDesc.renderState.depthStencilState.frontFaceStencil.passOp = nvrhi::StencilOp::Invert;
           // psoDesc.renderState.depthStencilState.backFaceStencil.stencilFunc = nvrhi::ComparisonFunc::Equal;
           // psoDesc.renderState.depthStencilState.backFaceStencil.passOp = nvrhi::StencilOp::Invert;
            psoDesc.bindingLayouts = { m_BindingLayout };
            psoDesc.inputLayout = m_InputLayout;
            psoDesc.renderState.rasterState.setFrontCounterClockwise(false);
            psoDesc.renderState.rasterState.fillMode = nvrhi::RasterFillMode::Wireframe;
            psoDesc.renderState.rasterState.setCullNone();

            m_PipelineWireFrame = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
            
            psoDesc.renderState.rasterState.fillMode = nvrhi::RasterFillMode::Fill;
            m_Pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
        }

        if (!m_BackgroundPSO)
        {
            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS = m_BackgroundVS;
            psoDesc.PS = m_BackgroundPS;
            psoDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
            psoDesc.renderState.depthStencilState.depthTestEnable = false;
            psoDesc.bindingLayouts = { m_BindingLayout };
           // psoDesc.inputLayout = m_InputLayout;
           // psoDesc.renderState.rasterState.setFrontCounterClockwise(true);
            psoDesc.renderState.rasterState.setCullNone();

            m_BackgroundPSO = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
        }

        m_CommandList->open();

        int2 windowSize;
        GetDeviceManager()->GetWindowDimensions(windowSize.x, windowSize.y);

        nvrhi::TextureHandle alphaTex = m_ommData.GetAlphaTexture();

        const float2 aspectRatioTex = float2((float)alphaTex->getDesc().width / (float)alphaTex->getDesc().height, 1.f);
        const float2 aspectRatioScreen = float2((float)windowSize.y / (float)windowSize.x, 1.f);

        auto GetTextureUvFromScreenPos = [this](const Constants& constants, float2 screenPos)
            {
                float2 uv = float2(screenPos.x, 1.f - screenPos.y);
                uv -= float2(0.5, 0.5);
                uv /= constants.zoom;
                uv /= constants.aspectRatio;
                uv += float2(0.5, 0.5);
                uv -= 0.5f * constants.offset;

                const int2 texelMousePos = (int2)donut::math::round(uv * (float2)constants.texSize);
                return texelMousePos;
            };

        Constants constants;
        constants.texSize = math::uint2(alphaTex->getDesc().width, alphaTex->getDesc().height);
        constants.invTexSize = float2(1.f / constants.texSize.x, 1.f / constants.texSize.y);
        constants.screenSize = float2(framebuffer->getFramebufferInfo().getViewport().width(), framebuffer->getFramebufferInfo().getViewport().height());
        constants.invScreenSize = 1.f / constants.screenSize;
        constants.zoom = m_ui.zoom;
        constants.offset = m_ui.offset + m_ui.prevOffset;
        constants.aspectRatio = aspectRatioTex * aspectRatioScreen;
        constants.primitiveOffset = m_ui.primitiveStart;
        constants.mode = 0;
        constants.ommIndexHighlight = m_ui.ommIndexHighlightEnable ? m_ui.ommIndexHighlight : -5;
        constants.ommIndexHighlightEnable = m_ui.ommIndexHighlightEnable;
        constants.ommIndexIsolate = m_ui.ommIndexIsolate;
        constants.drawAlphaContour = m_ui.drawAlphaContour;
        constants.colorizeStates = m_ui.colorizeStates;
        constants.alphaCutoff = m_ui.input.has_value() ? m_ui.input->alphaCutoff : -1.f;
        int2 mouseCoord = GetTextureUvFromScreenPos(constants, m_mousePos / (float2)(windowSize));
        constants.mouseCoordX = mouseCoord.x;
        constants.mouseCoordY = mouseCoord.y;
        m_CommandList->writeBuffer(m_ConstantBuffer, &constants, sizeof(constants));

        nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));

        int sizePerIndex = m_ommData.GetIndexBuffer()->getDesc().format == nvrhi::Format::R32_UINT ? 4 : 2;

        nvrhi::IndexBufferBinding indexBinding;
        indexBinding.buffer = m_ommData.GetIndexBuffer();
        indexBinding.format = m_ommData.GetIndexBuffer()->getDesc().format;
        indexBinding.offset = 3 * m_ui.primitiveStart * sizePerIndex;

        nvrhi::VertexBufferBinding vertexBinding;
        vertexBinding.buffer = m_ommData.GetTexCoordBuffer();
        vertexBinding.slot = 0;
        vertexBinding.offset = 0;

        {
            nvrhi::GraphicsState state;
            state.pipeline = m_BackgroundPSO;
            state.framebuffer = framebuffer;
            state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
            state.bindings = { m_BindingSets };
            m_CommandList->setGraphicsState(state);

            nvrhi::DrawArguments args;
            args.vertexCount = 4;
            m_CommandList->draw(args);
        }

        if (m_ui.drawMicroTriangles)
        {
            nvrhi::GraphicsState state;
            state.pipeline = m_Pipeline;
            state.framebuffer = framebuffer;
            state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
            state.bindings = { m_BindingSets };
            state.setIndexBuffer(indexBinding);
            state.addVertexBuffer(vertexBinding);
            m_CommandList->setGraphicsState(state);

            nvrhi::DrawArguments args;
            args.vertexCount = 3 * std::max(0, (m_ui.primitiveEnd - m_ui.primitiveStart));
            m_CommandList->drawIndexed(args);
        }

        constants.mode = 1;
        m_CommandList->writeBuffer(m_ConstantBuffer, &constants, sizeof(constants));
        
        if (m_ui.drawWireFrame)
        {
            nvrhi::GraphicsState state;
            state.pipeline = m_PipelineWireFrame;
            state.framebuffer = framebuffer;
            state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
            state.bindings = { m_BindingSets };
            state.setIndexBuffer(indexBinding);
            state.addVertexBuffer(vertexBinding);
            m_CommandList->setGraphicsState(state);

            nvrhi::DrawArguments args;
            args.vertexCount = 3 * std::max(0, (m_ui.primitiveEnd - m_ui.primitiveStart));
            m_CommandList->drawIndexed(args);
        }

        if (m_pixelReadback)
        {
            m_pixelReadback->Capture(m_CommandList, (donut::math::uint2)m_mousePos);
            auto val = m_pixelReadback->ReadFloats();
            m_ui.alphaVal = val.x;
            m_ui.texel.x = reinterpret_cast<int&>(val.y);
            m_ui.texel.y = reinterpret_cast<int&>(val.z);
            m_ui.ommIndexHighlight = reinterpret_cast<int&>(val.w) - 5;
        }

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
    }
};

template<class T>
void ImGui_CheckBoxFlag(const char* name, uint32_t ID, T& flags, T origFlags, T mask)
{
    bool value = ((uint32_t)mask & (uint32_t)flags) == (uint32_t)mask;
    bool origValue = ((uint32_t)mask & (uint32_t)origFlags) == (uint32_t)mask;
    
    ImGui::BeginDisabled(origValue == value);

    ImGui::PushID(ID);
    if (ImGui::Button("Reset"))
    {
        reinterpret_cast<uint32_t&>(flags) &= ~(uint32_t)mask;
        reinterpret_cast<uint32_t&>(flags) |= (uint32_t)origFlags & (uint32_t)mask;
    }
    ImGui::PopID();

    ImGui::EndDisabled();

    ImGui::SameLine();

    if (ImGui::Checkbox(name, &value))
    {
        if (value)
        {
            reinterpret_cast<uint32_t&>(flags) |= (uint32_t)mask;
        }
        else
        {
            reinterpret_cast<uint32_t&>(flags) &= ~(uint32_t)mask;
        }
    }
}

template<class T>
void ImGui_SliderInt(const char* name, uint32_t ID, T& value, T origValue, int min, int max)
{
    ImGui::BeginDisabled(origValue == value);

    ImGui::PushID(ID);
    if (ImGui::Button("Reset"))
    {
        value = origValue;
    }
    ImGui::PopID();

    ImGui::EndDisabled();

    ImGui::SameLine();

    int intVal = value;
    ImGui::SliderInt(name, &intVal, min, max);
    value = intVal;
}

bool ImGui_SliderFloat(const char* name, uint32_t ID, float& value, float origValue, float min, float max)
{
    ImGui::BeginDisabled(origValue == value);

    ImGui::PushID(ID);
    if (ImGui::Button("Reset"))
    {
        value = origValue;
    }
    ImGui::PopID();

    ImGui::EndDisabled();

    ImGui::SameLine();

    return ImGui::SliderFloat(name, &value, min, max);
}

void ImGui_ValueUInt64(const char* name, uint32_t ID, uint64_t& value, uint64_t origValue)
{
    ImGui::BeginDisabled(origValue == value);

    ImGui::PushID(ID);
    if (ImGui::Button("Reset"))
    {
        value = origValue;
    }
    ImGui::PopID();

    ImGui::EndDisabled();

    ImGui::SameLine();

    ImGuiInputTextFlags flags = ImGuiInputTextFlags_CharsHexadecimal;
    const char* format = (flags & ImGuiInputTextFlags_CharsHexadecimal) ? "%llX" : "%llu";
    int step = 1;
    int step_fast = 100;
    ImGui::InputScalar(name, ImGuiDataType_U64, (void*)&value, (void*)(step > 0 ? &step : NULL), (void*)(step_fast > 0 ? &step_fast : NULL), format, flags);
}

template<class T, int N>
void ImGui_Combo(const char* name, uint32_t ID, const std::array<const char*, N>& itemNames, const std::array<T, N>& itemValues, T& value, T origValue)
{
    ImGui::BeginDisabled(origValue == value);

    ImGui::PushID(ID);
    if (ImGui::Button("Reset"))
    {
        value = origValue;
    }
    ImGui::PopID();

    ImGui::EndDisabled();

    ImGui::SameLine();

    int selectedIndex = 0;
    for (auto& item : itemValues)
    {
        if (item == value)
        {
            break;
        }
        selectedIndex++;
    }

    if (ImGui::Combo(name, reinterpret_cast<int*>(&selectedIndex), itemNames.data(), (int)itemNames.size()))
    {
        value = itemValues[selectedIndex];
    }
}

class UIRenderer : public donut::app::ImGui_Renderer
{
private:
    std::unique_ptr<donut::app::ImGui_Console> m_console;
    UIData& m_ui;
    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    std::shared_ptr<BasicTriangle> m_app;
    std::shared_ptr<donut::app::RegisteredFont> m_FontOpenSans;
    // create a file browser instance
    ImGui::FileBrowser fileDialog;

public:
    ~UIRenderer()
    {

    }
    UIRenderer(donut::app::DeviceManager* deviceManager, std::shared_ptr<BasicTriangle> app, UIData& ui)
        : ImGui_Renderer(deviceManager)
        , m_app(app)
        , m_ui(ui)
        , fileDialog(ImGuiFileBrowserFlags_SelectDirectory | ImGuiFileBrowserFlags_NoModal | ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_ConfirmOnEnter)
    {
        auto rootFs = std::make_shared<vfs::RootFileSystem>();

        float scaleX, scaleY;
        GetDeviceManager()->GetDPIScaleInfo(scaleX, scaleY);

        m_FontOpenSans = CreateFontFromMemoryCompressed((void*)OpenSans_compressed_data, sizeof(OpenSans_compressed_data), 17.f);
        ImGui::GetStyle().ScaleAllSizes(scaleX);

        // (optional) set browser properties
        fileDialog.SetTitle("Select directory of bake input binaries to view (.bin)");
        fileDialog.SetTypeFilters({ ".bin" });

        m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFs, "/shaders");
        // ImGui::GetIO().IniFilename = "ui_state.ini";

        ImGui::LoadIniSettingsFromDisk(ImGui::GetIO().IniFilename);

        if (fileDialog.HasSelected())
        {
            SelectFileDir(fileDialog.GetSelected().string(), -1);
            fileDialog.ClearSelected();
        }
        else {

            std::string defaultPath = OMM_VIEWER_DEFAULT_BINARY_FOLDER;
            int fileIndex = -1;
            std::ifstream infile;
            infile.open("ui_dir_state.ini", std::ios_base::in);
            if (infile.is_open())
            {
                std::string dir;
                infile >> dir;
                infile >> fileIndex;
                infile.close();

                if (std::filesystem::exists(dir))
                {
                    SelectFileDir(dir, fileIndex);
                }
                else
                {
                    SelectFileDir(defaultPath, fileIndex);
                }
            }
            else
            {
                SelectFileDir(defaultPath, fileIndex);
            }
        }
    }

    void Init()
    {
        donut::app::ImGui_Renderer::Init(m_ShaderFactory);
    }

protected:

    void SelectFileDir(const std::string& dir, int fileIndex)
    {
        if (!fileDialog.HasSelected())
        {
            fileDialog.SetDirectory(dir);
        }

        m_ui.ommFiles.clear();
        m_ui.path = dir;

        std::ofstream outfile;
        outfile.open("ui_dir_state.ini", std::ios_base::trunc);
        if (outfile.is_open())
        {
            outfile << dir << std::endl;
            outfile << fileIndex;
            outfile.close();
        }

        if (!std::filesystem::exists(m_ui.path))
            return;

        for (const auto& entry : std::filesystem::directory_iterator(m_ui.path))
        {
            auto ext = entry.path().extension();
            if (ext == ".bin")
                m_ui.ommFiles.push_back(entry.path());
        }

        std::sort(m_ui.ommFiles.begin(), m_ui.ommFiles.end(), [](const std::filesystem::path& a, const std::filesystem::path& b) {
            return a.filename().string() < b.filename().string();
            });

        if (m_ui.ommFiles.size() != 0)
        {
            m_ui.selectedFile = std::clamp<int>(fileIndex, 0, (int)m_ui.ommFiles.size() - 1);
            m_ui.rebake = true;
        }
        else
        {
            m_ui.selectedFile = -1;
        }

        m_ui.load = true;
    }

    ImVec4 sRGBToLinear(const ImVec4& color) {
        auto linearize = [](float c) {
            return (c <= 0.04045f) ? c / 12.92f : pow((c + 0.055f) / 1.055f, 2.4f);
            };
        return ImVec4(linearize(color.x), linearize(color.y), linearize(color.z), color.w);
    }

    void SetStyle()
    {
            ImGuiStyle& style = ImGui::GetStyle();
            // Define the primary NVIDIA green color
           // ImVec4 nvidiaGreen = ImVec4(0.46f, 0.73f, 0.00f, 1.00f); // #76B900
            ImVec4 nvidiaGreen = sRGBToLinear(ImVec4(0.46f, 0.73f, 0.00f, 1.00f));
            
            // Darken background colors and adjust other colors for contrast
            style.Colors[ImGuiCol_Text] = ImVec4(0.85f, 0.88f, 0.85f, 1.00f);  // Light grey text
            style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.55f, 0.50f, 1.00f);  // Dark grey for disabled text
            style.Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);  // Very dark background
            style.Colors[ImGuiCol_Border] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);  // Darker border
            style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);  // No shadow

            // Frame and background colors
            style.Colors[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);  // Darker grey frame background
            style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);  // Slightly lighter frame on hover
            style.Colors[ImGuiCol_FrameBgActive] = nvidiaGreen;                         // NVIDIA green for active frame

            // Title bar colors
            style.Colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);  // Very dark title background
            style.Colors[ImGuiCol_TitleBgActive] = nvidiaGreen;                         // Green for active title background
            style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.08f, 0.08f, 0.08f, 0.75f);  // Slight transparency when collapsed

            // Scrollbars and sliders
            style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);  // Dark scrollbar background
            style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);  // Darker grey for scrollbar grab
            style.Colors[ImGuiCol_ScrollbarGrabHovered] = nvidiaGreen;                         // Green when hovered
            style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.36f, 0.53f, 0.00f, 1.00f);  // Darker green when active

            // Buttons
            style.Colors[ImGuiCol_Button] = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);  // Dark button
            style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.36f, 0.53f, 0.00f, 1.00f);  // Hovered button in darker green
            style.Colors[ImGuiCol_ButtonActive] = nvidiaGreen;                         // Active button in NVIDIA green

            // Headers (for collapsible headers, etc.)
            style.Colors[ImGuiCol_Header] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);  // Dark header
            style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);  // Lighter on hover
            style.Colors[ImGuiCol_HeaderActive] = nvidiaGreen;                         // Active header in green

            // Other interactive elements
            style.Colors[ImGuiCol_CheckMark] = nvidiaGreen;                         // Checkmark in NVIDIA green
            style.Colors[ImGuiCol_SliderGrab] = nvidiaGreen;                         // Slider in NVIDIA green
            style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.36f, 0.53f, 0.00f, 1.00f);  // Darker green for active slider
            style.Colors[ImGuiCol_Separator] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);  // Darker separator

            // Plot colors
            style.Colors[ImGuiCol_PlotLines] = nvidiaGreen;                         // Lines in green
            style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.36f, 0.53f, 0.00f, 1.00f);  // Darker green for hover
            style.Colors[ImGuiCol_PlotHistogram] = nvidiaGreen;                         // Histogram in green
            style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.36f, 0.53f, 0.00f, 1.00f);  // Darker green when hovered

            // Selection and popup backgrounds
            style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.36f, 0.53f, 0.00f, 0.50f);  // Semi-transparent green selection
            style.Colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.95f);  // Very dark background for popups

    }

    virtual void buildUI(void) override
    {
        if (!m_ui.ShowUI)
            return;

        const auto& io = ImGui::GetIO();
        
        SetStyle();
        if (m_FontOpenSans)
            ImGui::PushFont(m_FontOpenSans->GetScaledFont());

        int2 windowSize;
        GetDeviceManager()->GetWindowDimensions(windowSize.x, windowSize.y);

        float scaleX = 1.f, scaleY = 1.f;
        GetDeviceManager()->GetDPIScaleInfo(scaleX, scaleY);

        ImGui::SetNextWindowBgAlpha(0.98f);
        ImGui::SetNextWindowPos(ImVec2(windowSize.x - scaleX * 155, windowSize.y - scaleY * 80.f));
        ImGui::SetNextWindowSizeConstraints(ImVec2(10.f, 10.f), ImVec2(windowSize.x - 20.f, windowSize.y - 20.f));

        ImGui::Begin("Info", 0, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoTitleBar);
        ImGui::Text("Alpha:%.6f", m_ui.alphaVal);
        ImGui::Text("Texel:(%d, %d)", m_ui.texel.x, m_ui.texel.y);
        if (const omm::Cpu::BakeResultDesc* result = m_app->GetOmmGpuData().GetResult())
        {
            if (m_ui.ommIndexHighlight >= 0 && (uint32_t)m_ui.ommIndexHighlight < result->descArrayCount)
            {
                omm::Cpu::OpacityMicromapDesc desc = result->descArray[m_ui.ommIndexHighlight];
                ImGui::Text("Desc Index:(%d), lvl:(%d)", m_ui.ommIndexHighlight, desc.subdivisionLevel);
            }
        }
        ImGui::End();

        ImGui::SetNextWindowBgAlpha(0.98f);
        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSizeConstraints(ImVec2(10.f, 10.f), ImVec2(windowSize.x - 20.f, windowSize.y - 20.f));

        ImGui::Begin("Settings", 0, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar);

        ImGui::Text("Renderer: %s", GetDeviceManager()->GetRendererString());
        double frameTime = GetDeviceManager()->GetAverageFrameTimeSeconds();
        if (frameTime > 0.0)
            ImGui::Text("%.3f ms/frame (%.1f FPS)", frameTime * 1e3, 1.0 / frameTime);

        int maxPrimitiveCount = m_app->GetOmmGpuData().GetIndexCount() / 3;

        if (!m_ui.path.empty())
        {
            if (ImGui::Button(m_ui.path.c_str()))
                fileDialog.Open();
        }
        else
        {
            if (ImGui::Button("Select a path with .bin files"))
                fileDialog.Open();
        }

        fileDialog.Display();

        if (fileDialog.HasSelected())
        {
            SelectFileDir(fileDialog.GetSelected().string(), -1);
            fileDialog.ClearSelected();
        }

        ImGui::SameLine();
        ImGui::Text("Path");

        if (m_ui.ommFiles.size() != 0)
        {
            auto& files = m_ui.ommFiles;
            auto selected = files[m_ui.selectedFile].filename().string();
            bool updateFileDir = false;
            if (ImGui::BeginCombo("Data", selected.c_str())) // Pass in the label and the current item
            {
                for (int i = 0; i < files.size(); i++)
                {
                    auto file = files[i].filename().string();

                    bool is_selected = (m_ui.selectedFile == i);
                    if (ImGui::Selectable(file.c_str(), is_selected))
                    {
                        updateFileDir = true;
                        m_ui.selectedFile = i;
                    }
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            if (updateFileDir)
                SelectFileDir(m_ui.path, m_ui.selectedFile);
        }
        else
        {
            ImGui::Text("Path contains no .bin files");
        }

        ImGui::BeginDisabled(!m_app->GetOmmGpuData().HasLoadedData());

        if (m_app->GetOmmGpuData().HasLoadedData() && m_ui.input.has_value())
        {
            if (m_ui.primitiveEnd == -1)
            {
                m_ui.primitiveEnd = maxPrimitiveCount;// std::min(128, maxPrimitiveCount);
            }

            if (ImGui::SliderInt("Primitive Start", &m_ui.primitiveStart, 0, maxPrimitiveCount - 1, "%d"))
            {
                if (m_ui.primitiveStart >= m_ui.primitiveEnd)
                    m_ui.primitiveEnd = m_ui.primitiveStart + 1;
            }

            if (ImGui::SliderInt("Primitive End", &m_ui.primitiveEnd, 1, maxPrimitiveCount, "%d"))
            {
                if (m_ui.primitiveStart >= m_ui.primitiveEnd)
                    m_ui.primitiveStart = m_ui.primitiveEnd - 1;
            }

            int ommIndexCount = m_app->GetOmmGpuData().GetOmmIndexCount();
            ImGui::SliderInt("Isolate OMM Desc", &m_ui.ommIndexIsolate, -1, ommIndexCount, "%d");

            const omm::Cpu::TextureDesc& texDesc = m_app->GetOmmGpuData().GetDefaultTextureDesc();
            const omm::Cpu::BakeInputDesc& input = m_app->GetOmmGpuData().GetDefaultInput();
            
            ImGui::SeparatorText("Memory");

            if (const omm::Cpu::BakeResultDesc* result = m_app->GetOmmGpuData().GetResult())
            {
                size_t arrayDataSize = result->arrayDataSize;
                size_t indexSize;
                if (result->indexFormat == omm::IndexFormat::UINT_8)
                    indexSize = result->indexCount * 1;
                else if (result->indexFormat == omm::IndexFormat::UINT_16)
                    indexSize = result->indexCount * 2;
                else // omm::IndexFormat::UINT_32
                    indexSize = result->indexCount * 4;
                size_t descArraySize = result->descArrayCount * sizeof(omm::Cpu::OpacityMicromapDesc);
                size_t totalSize = arrayDataSize + indexSize + descArraySize;
                ImGui::Text("Array Data Size %.4f mb (%zu bytes)", arrayDataSize / (1024.f * 1024.f), arrayDataSize);
                ImGui::Text("Index Data Size %.4f mb (%zu bytes)", indexSize / (1024.f * 1024.f), indexSize);
                ImGui::Text("Desc Array Size %.4f mb (%zu bytes)", descArraySize / (1024.f * 1024.f), descArraySize);
                ImGui::Text("Total Size %.4f mb (%zu bytes)", totalSize / (1024.f * 1024.f), totalSize);
            }

            ImGui::SeparatorText("Stats");

            omm::Debug::Stats stats = m_app->GetOmmGpuData().GetStats();
            const float known = (float)stats.totalOpaque + stats.totalTransparent;
            const float unknown = (float)stats.totalUnknownTransparent + stats.totalUnknownOpaque;

            if (const omm::Cpu::BakeResultDesc* result = m_app->GetOmmGpuData().GetResult())
            {
                const float indexCount = (float)result->indexCount;
                const float descCount = (float)result->descArrayCount;

                ImGui::Text("Tri per block: %.2f", indexCount / descCount);
            }

            float total = known + unknown;
            ImGui::Text("Known %.2f%%", 100.f * known / (known + unknown));
            ImGui::Text("Known Area %.2f%%", 100.f * stats.knownAreaMetric);
            ImGui::Text("Total Opaque %lu (%.2f%%)", stats.totalOpaque, 100.f * stats.totalOpaque / total);
            ImGui::Text("Total Transparent %lu (%.2f%%)", stats.totalTransparent, 100.f * stats.totalTransparent / total);
            ImGui::Text("Total Unknown Transparent %lu (%.2f%%)", stats.totalUnknownTransparent, 100.f * stats.totalUnknownTransparent / total);
            ImGui::Text("Total Unknown Opaque %lu (%.2f%%)", stats.totalUnknownOpaque, 100.f * stats.totalUnknownOpaque / total);

            ImGui::Text("Total Fully Opaque %u", stats.totalFullyOpaque);
            ImGui::Text("Total Fully Transparent %u", stats.totalFullyTransparent);
            ImGui::Text("Total Fully Unknown Transparent %u", stats.totalFullyUnknownTransparent);
            ImGui::Text("Total Fully Unknown Opaque %u", stats.totalFullyUnknownOpaque);

            if (ImGui::CollapsingHeader("Bake Settings", ImGuiTreeNodeFlags_DefaultOpen))
            {
                uint32_t id = 0;

                ImGui::SeparatorText("Texture Settings");

                if (nvrhi::ITexture* texture = m_app->GetOmmGpuData().GetAlphaTexture())
                {
                    uint width = texture->getDesc().width;
                    uint height = texture->getDesc().height;
                    nvrhi::Format format = texture->getDesc().format;

                    const char* formatstr = "Format unknown";
                    if (format == nvrhi::Format::R32_FLOAT)
                        formatstr = "Float 32";
                    else if (format == nvrhi::Format::R8_UNORM)
                        formatstr = "Unorm 8";

                    ImGui::Text("Alpha Texture %dx%d,%s", width, height, formatstr);
                }

                ImGui_Combo<omm::TextureAddressMode, 5>("Addressing Mode", id++,
                    {
                      "Wrap",
                      "Mirror",
                      "Clamp",
                      "Border",
                      "MirrorOnce"
                    },
                {
                  omm::TextureAddressMode::Wrap,
                  omm::TextureAddressMode::Mirror,
                  omm::TextureAddressMode::Clamp,
                  omm::TextureAddressMode::Border,
                  omm::TextureAddressMode::MirrorOnce
                },
                    m_ui.input->runtimeSamplerDesc.addressingMode, input.runtimeSamplerDesc.addressingMode);

                ImGui_CheckBoxFlag<omm::Cpu::TextureFlags>("Disable Z Order", id++, m_ui.textureDesc->flags, texDesc.flags, omm::Cpu::TextureFlags::DisableZOrder);

                {
                    ImGui::BeginDisabled(texDesc.alphaCutoff == m_ui.textureDesc->alphaCutoff);

                    ImGui::PushID(id++);
                    if (ImGui::Button("Reset"))
                    {
                        m_ui.textureDesc->alphaCutoff = texDesc.alphaCutoff;
                    }
                    ImGui::PopID();

                    ImGui::EndDisabled();

                    ImGui::SameLine();

                    bool value = m_ui.textureDesc->alphaCutoff >= 0.f;
                    if (ImGui::Checkbox("Enable SAT acceleration", &value))
                    {
                        if (value)
                        {
                            m_ui.textureDesc->alphaCutoff = input.alphaCutoff;
                        }
                        else
                        {
                            m_ui.textureDesc->alphaCutoff = -1.f;
                        }
                    }
                }

                ImGui::SeparatorText("Bake Settings");

                if (ImGui_SliderFloat("Alpha Cutoff", id++, m_ui.input->alphaCutoff, input.alphaCutoff, 0.f, 1.f))
                {
                    if (m_ui.textureDesc->alphaCutoff >= 0.f)
                        m_ui.textureDesc->alphaCutoff = m_ui.input->alphaCutoff;
                }

                ImGui_Combo<omm::Format, 2>("Format", id++,
                    {
                        "OC1_2_State",
                        "OC1_4_State",
                    },
                {
                    omm::Format::OC1_2_State,
                    omm::Format::OC1_4_State
                },
                    m_ui.input->format, input.format);

                ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Enable Internal Threads", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::EnableInternalThreads);
                ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Disable Special Indices", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::DisableSpecialIndices);
                ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Force 32 Bit Indices", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::Force32BitIndices);
                ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Disable Duplicate Detection", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::DisableDuplicateDetection);
                ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Enable Near-Duplicate Detection", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::EnableNearDuplicateDetection);
                ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Enable Validation", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::EnableValidation);

                ImGui_SliderInt<uint8_t>("Max Subdivision Level", id++, m_ui.input->maxSubdivisionLevel, input.maxSubdivisionLevel, 0, 12);
                {
                    ImGui::BeginDisabled(input.dynamicSubdivisionScale == m_ui.input->dynamicSubdivisionScale);

                    ImGui::PushID(id++);
                    if (ImGui::Button("Reset"))
                    {
                        m_ui.input->dynamicSubdivisionScale = input.dynamicSubdivisionScale;
                    }
                    ImGui::PopID();
                    ImGui::EndDisabled();

                    ImGui::PushID(id++);
                    ImGui::SameLine();
                    if (ImGui::Button("-1.f"))
                    {
                        m_ui.input->dynamicSubdivisionScale = -1.f;
                    }
                    ImGui::PopID();

                    ImGui::SameLine();

                    ImGui::SliderFloat("Dynamic Subdivision Scale", &m_ui.input->dynamicSubdivisionScale, 0.f, 100.f, "%.3f", ImGuiSliderFlags_Logarithmic);
                }

                {
                    ImGui::BeginDisabled(input.maxArrayDataSize == m_ui.input->maxArrayDataSize);

                    ImGui::PushID(id++);
                    if (ImGui::Button("Reset"))
                    {
                        m_ui.input->maxArrayDataSize = input.maxArrayDataSize;
                    }
                    ImGui::PopID();
                    ImGui::EndDisabled();

                    ImGui::PushID(id++);
                    ImGui::SameLine();
                    if (ImGui::Button("Disable"))
                    {
                        m_ui.input->maxArrayDataSize = -1;
                    }
                    ImGui::PopID();

                    ImGui::PushID(id++);
                    ImGui::SameLine();
                    if (ImGui::Button("Current"))
                    {
                        const omm::Cpu::BakeResultDesc* result = m_app->GetOmmGpuData().GetResult();
                        m_ui.input->maxArrayDataSize = result ? result->arrayDataSize : 0;
                    }
                    ImGui::PopID();

                    ImGui::PushID(id++);
                    ImGui::SameLine();
                    if (ImGui::Button("x0.5"))
                    {
                        m_ui.input->maxArrayDataSize = m_ui.input->maxArrayDataSize / 2;
                    }
                    ImGui::PopID();

                    ImGui::PushID(id++);
                    ImGui::SameLine();
                    if (ImGui::Button("x2"))
                    {
                        m_ui.input->maxArrayDataSize = m_ui.input->maxArrayDataSize * 2;
                    }
                    ImGui::PopID();

                    ImGui::SameLine();

                    ImGui::SliderInt("Target Memory", reinterpret_cast<int*>(&m_ui.input->maxArrayDataSize), 0, 1000000,"%d", ImGuiSliderFlags_Logarithmic);
                }

                ImGui_SliderFloat("Rejection Threshold", id++, m_ui.input->rejectionThreshold, input.rejectionThreshold, 0.f, 1.f);
                ImGui_SliderFloat("Near Duplicate Deduplication Factor", id++, m_ui.input->nearDuplicateDeduplicationFactor, input.nearDuplicateDeduplicationFactor, 0.f, 1.f);
                
                ImGui::BeginDisabled(m_ui.input->format == omm::Format::OC1_4_State);
                ImGui_Combo<omm::UnknownStatePromotion, 3>("Unknown State Promotion", id++,
                    {
                        "Nearest",
                        "Force Opaque",
                        "Force Transparent",
                    },
                {
                    omm::UnknownStatePromotion::Nearest,
                    omm::UnknownStatePromotion::ForceOpaque,
                    omm::UnknownStatePromotion::ForceTransparent,
                },
                m_ui.input->unknownStatePromotion, input.unknownStatePromotion);
                ImGui::EndDisabled();

                ImGui_ValueUInt64("Max Workload Size", id++, m_ui.input->maxWorkloadSize, input.maxWorkloadSize);
                ImGui::SeparatorText("Unofficial Bake Settings");

                constexpr uint32_t kEnableAABBTesting = 1u << 6u;
                constexpr uint32_t kDisableLevelLineIntersection = 1u << 7u;
                constexpr uint32_t kDisableFineClassification = 1u << 8u;
                constexpr uint32_t kEnableNearDuplicateDetectionBruteForce = 1u << 9u;
                constexpr uint32_t kEdgeHeuristic = 1u << 10u;
                ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Enable AABB Testing", id++, m_ui.input->bakeFlags, input.bakeFlags, (omm::Cpu::BakeFlags)kEnableAABBTesting);
                ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Disable Level Line Intersection", id++, m_ui.input->bakeFlags, input.bakeFlags, (omm::Cpu::BakeFlags)kDisableLevelLineIntersection);
                ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Disable Fine Classification", id++, m_ui.input->bakeFlags, input.bakeFlags, (omm::Cpu::BakeFlags)kDisableFineClassification);
                ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Enable Near-Duplicate Detection Brute-Force", id++, m_ui.input->bakeFlags, input.bakeFlags, (omm::Cpu::BakeFlags)kEnableNearDuplicateDetectionBruteForce);
                ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Edge Heuristic", id++, m_ui.input->bakeFlags, input.bakeFlags, (omm::Cpu::BakeFlags)kEdgeHeuristic);

                ImGui::Separator();

                if (ImGui::Button("Rebake"))
                {
                    m_ui.rebake = true;
                }

                ImGui::SameLine();
                ImGui::Text("Last bake time %lus, (%lu ms)", m_app->GetOmmGpuData().GetBakeTimeInSeconds(), m_app->GetOmmGpuData().GetBakeTimeInMs());
            }
           
            if (ImGui::CollapsingHeader("Histogram"))
            {
                if (const omm::Cpu::BakeResultDesc* result = m_app->GetOmmGpuData().GetResult())
                {
                    {
                        std::array<float, 12> histogramOC2 = { 0, }, histogramOC4 = { 0, };
                        for (uint32_t i = 0; i < result->descArrayHistogramCount; ++i)
                        {
                            if ((omm::Format)result->descArrayHistogram[i].format == omm::Format::OC1_2_State)
                            {
                                uint32_t count = result->descArrayHistogram[i].count;
                                histogramOC2[result->descArrayHistogram[i].subdivisionLevel] = (float)count;
                            }

                            if ((omm::Format)result->descArrayHistogram[i].format == omm::Format::OC1_4_State)
                            {
                                uint32_t count = result->descArrayHistogram[i].count;
                                histogramOC4[result->descArrayHistogram[i].subdivisionLevel] = (float)count;
                            }
                        }

                        ImGui::PlotHistogram("", histogramOC2.data(), (int)histogramOC2.size(), 0, "Desc Histogram (OC2)", 0.f, 12.f, ImVec2(0, 80));
                        ImGui::SameLine();
                        ImGui::PlotHistogram("", histogramOC4.data(), (int)histogramOC4.size(), 0, "Desc Histogram (OC4)", 0.f, 12.f, ImVec2(0, 80));
                    }
                }
            }

            if (ImGui::CollapsingHeader("Visualiztion"))
            {
                ImGui::Checkbox("Draw Alpha Contour", &m_ui.drawAlphaContour);
                ImGui::Checkbox("Draw Wire-Frame", &m_ui.drawWireFrame);
                ImGui::Checkbox("Draw Micro-Triangles", &m_ui.drawMicroTriangles);
                ImGui::Checkbox("Colorize States", &m_ui.colorizeStates);
                ImGui::Checkbox("Enable OMM Index Highlight", &m_ui.ommIndexHighlightEnable);
            }
        }
       
        ImGui::EndDisabled();

        if (ImGui::CollapsingHeader("Development")) {
            if (ImGui::Button("Recompile Shaders"))
            {
                m_ui.recompile = true;
            }
        }
        ImGui::End();

        if (m_FontOpenSans)
            ImGui::PopFont();
    }
};

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true; 
    deviceParams.enableNvrhiValidationLayer = true;
#endif
    deviceParams.enablePerMonitorDPI = true;
    deviceParams.backBufferWidth = 2 * 1280;
    deviceParams.backBufferHeight = 2 * 720;
    deviceParams.vsyncEnabled = true;
    deviceParams.supportExplicitDisplayScaling = true;
#if DONUT_WITH_VULKAN
    deviceParams.requiredVulkanDeviceExtensions.push_back(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
    deviceParams.requiredVulkanDeviceExtensions.push_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
    deviceParams.requiredVulkanDeviceExtensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
#endif

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    
    {
        UIData ui;
        std::shared_ptr<BasicTriangle> example = std::make_shared<BasicTriangle>(deviceManager, ui);
        example->Init();
        std::shared_ptr<UIRenderer> gui = std::make_shared<UIRenderer>(deviceManager, example, ui);
        gui->Init();

        {
            deviceManager->AddRenderPassToBack(example.get());
            deviceManager->AddRenderPassToBack(gui.get());

            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(example.get());
            deviceManager->RemoveRenderPass(gui.get());
        }
    }
    
    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
