/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "omm.h"
#include "omm_handle.h"

#include "std_containers.h"
#include "log.h"

#include "util/math.h"
#include "util/assert.h"
#include "util/bit_tricks.h"
#include "util/texture.h"

namespace omm
{
    enum class TilingMode {
        Linear,
        MortonZ,
        MAX_NUM,
    };

    class TextureImpl
    {
    public:
        static inline constexpr HandleType kHandleType = HandleType::Texture;

        TextureImpl(const StdAllocator<uint8_t>& stdAllocator, const Logger& log);
        ~TextureImpl();

        ommResult Create(const ommCpuTextureDesc& desc);

        template<ommCpuTextureFormat eFormat, TilingMode eTilingMode>
        float Load(const int2& texCoord, int32_t mip) const;

        float Load(const int2& texCoord, int32_t mip) const;

        template<ommCpuTextureFormat eFormat, TilingMode eTilingMode, ommTextureAddressMode eMode, bool bTexIsPow2>
        float Bilinear(const float2& p, int32_t mip) const;

        float Bilinear(ommTextureAddressMode mode, const float2& p, int32_t mip) const;

        ommResult GetTextureDesc(ommCpuTextureDesc& desc) const;

        TilingMode GetTilingMode() const {
            return m_tilingMode;
        }
        
        ommCpuTextureFormat GetTextureFormat() const {
            return m_textureFormat;
        }

        const int2& GetSize(int32_t mip) const {
            return m_mips[mip].size;
        }

        const int2& GetSizeLog2(int32_t mip) const {
            return m_mips[mip].sizeLog2;
        }

        const float2& GetSizef(int32_t mip) const {
            return m_mips[mip].sizef;
        }

        bool SizeIsPow2() const {
            return m_mips[0].sizeIsPow2;
        }

        const float2& GetRcpSize(int32_t mip) const {
            return m_mips[mip].rcpSize;
        }

        uint32_t GetMipCount() const {
            return (uint32_t)m_mips.size();
        }

        bool HasAlphaCutoff() const {
            return m_alphaCutoff >= 0.f;
        }

        void SetAlphaCutoff(float alphaCutoff) {
            m_alphaCutoff = alphaCutoff;
        }
        float GetAlphaCutoff() const {
            return m_alphaCutoff;
        }

        bool InTexture(int2 texCoord, int32_t mip) const
        {
            return texCoord.x >= 0 &&
                texCoord.y >= 0 &&
                texCoord.x < m_mips[mip].size.x &&
                texCoord.y < m_mips[mip].size.y;
        }

        bool HasSAT() const
        {
            return m_dataSAT != nullptr;
        }

        uint32_t SAT(int2 s, int2 e, int32_t mip) const
        {
            OMM_ASSERT(InTexture(s, mip));
            OMM_ASSERT(InTexture(e, mip));
            uint32_t* dataSAT = (uint32_t*)(m_dataSAT + m_mips[mip].dataOffsetSAT);

            int32_t s_x_minus_one = (s.x - 1);
            int32_t s_y_minus_one = (s.y - 1);

            const uint32_t A = s_x_minus_one >= 0 && s_y_minus_one >= 0 ? dataSAT[(s_x_minus_one) + (s_y_minus_one) * m_mips[mip].size.x] : 0;
            const uint32_t B = s_y_minus_one >= 0 ? dataSAT[e.x + (s_y_minus_one) * m_mips[mip].size.x] : 0;
            const uint32_t C = s_x_minus_one >= 0 ? dataSAT[(s_x_minus_one) + (e.y) * m_mips[mip].size.x] : 0;
            const uint32_t D = dataSAT[e.x + (e.y) * m_mips[mip].size.x];
            int32_t sum = D + A - B - C;
            return sum;
        }

        template<class TMemoryStreamBuf>
        void Serialize(TMemoryStreamBuf& buffer) const;

        template<class TMemoryStreamBuf>
        void Deserialize(TMemoryStreamBuf& buffer, int inputDescVersion);

    private:

        ommResult Validate(const ommCpuTextureDesc& desc) const;
        void Deallocate();
        template<TilingMode eTilingMode>
        static uint32_t From2Dto1D(const int2& idx, const int2& size) {
            OMM_ASSERT(false && "Not implemented");
            return 0;
        }
        template<TilingMode eTilingMode>
        static uint2 From1Dto2D(const uint32_t idx, const int2& size) {
            OMM_ASSERT(false && "Not implemented");
            return uint2(0, 0);
        }
    private:
        static inline uint2  kMaxDim = int2(65536);
        static constexpr size_t kAlignment = 64;

        StdAllocator<uint8_t> m_stdAllocator;
        const Logger& m_log;

        struct Mips
        {
            int2 size;
            int2 sizeLog2;
            float2 sizef;
            bool sizeIsPow2;
            float2 rcpSize;
            int2 sizeMinusOne;
            uintptr_t dataOffset;
            size_t numElements;
            uintptr_t dataOffsetSAT;
        };

        vector<Mips> m_mips;
        TilingMode m_tilingMode;
        ommCpuTextureFormat m_textureFormat;
        ommCpuTextureFlags m_textureFlags;
        float m_alphaCutoff;
        uint8_t* m_data;
        size_t m_dataSize;
        uint8_t* m_dataSAT;
        size_t m_dataSATSize;
    };

    template<ommCpuTextureFormat eFormat, TilingMode eTilingMode>
    float TextureImpl::Load(const int2& texCoord, int32_t mip) const
    {
        OMM_ASSERT(eTilingMode == m_tilingMode);
        OMM_ASSERT(eFormat == m_textureFormat);
        OMM_ASSERT(texCoord.x >= 0);
        OMM_ASSERT(texCoord.y >= 0);
        OMM_ASSERT(texCoord.x < m_mips[mip].size.x);
        OMM_ASSERT(texCoord.y < m_mips[mip].size.y);
        OMM_ASSERT(texCoord.y < m_mips[mip].size.y);
        OMM_ASSERT(glm::all(glm::notEqual(texCoord, kTexCoordBorder2)));
        OMM_ASSERT(glm::all(glm::notEqual(texCoord, kTexCoordInvalid2)));
        const uint64_t idx = From2Dto1D<eTilingMode>(texCoord, m_mips[mip].size);
        OMM_ASSERT(idx < m_mips[mip].numElements);

        if constexpr (eFormat == ommCpuTextureFormat_FP32)
            return ((float*)(m_data + m_mips[mip].dataOffset))[idx];
        else if constexpr (eFormat == ommCpuTextureFormat_UNORM8)
            return (float)((uint8_t*)(m_data + m_mips[mip].dataOffset))[idx] * (1.f / 255.f);
        else
        {
            assert(false);
            return 0;
        }
    }


    template<ommCpuTextureFormat eFormat, TilingMode eTilingMode, ommTextureAddressMode eMode, bool bTexIsPow2>
    float TextureImpl::Bilinear(const float2& p, int32_t mip) const
    {
        float2 pixel = p * (float2)(m_mips[mip].size) - 0.5f;
        float2 pixelFloor = glm::floor(pixel);
        int2 coords[omm::TexelOffset::MAX_NUM];
        omm::GatherTexCoord4<eMode, bTexIsPow2>(int2(pixelFloor), m_mips[mip].size, coords);

        float a = Load<eFormat, eTilingMode>(coords[omm::TexelOffset::I0x0], mip);
        float b = Load<eFormat, eTilingMode>(coords[omm::TexelOffset::I0x1], mip);
        float c = Load<eFormat, eTilingMode>(coords[omm::TexelOffset::I1x0], mip);
        float d = Load<eFormat, eTilingMode>(coords[omm::TexelOffset::I1x1], mip);

        const float2 weight = glm::fract(pixel);
        float ac = glm::lerp<float>(a, c, weight.x);
        float bd = glm::lerp<float>(b, d, weight.x);
        float bilinearValue = glm::lerp(ac, bd, weight.y);
        return bilinearValue;
    }

   	template<> uint32_t TextureImpl::From2Dto1D<TilingMode::Linear>(const int2& idx, const int2& size);
   	template<> uint32_t TextureImpl::From2Dto1D<TilingMode::MortonZ>(const int2& idx, const int2& size);

    template<> uint2 TextureImpl::From1Dto2D<TilingMode::Linear>(const uint32_t idx, const int2& size);
    template<> uint2 TextureImpl::From1Dto2D<TilingMode::MortonZ>(const uint32_t idx, const int2& size);


    template<class TMemoryStreamBuf>
    void TextureImpl::Serialize(TMemoryStreamBuf& buffer) const
    {
        std::ostream os(&buffer);

        int numMips = (int)m_mips.size();
        os.write(reinterpret_cast<const char*>(&numMips), sizeof(numMips));

        if (numMips != 0)
        {
            for (const auto& mip : m_mips)
            {
                os.write(reinterpret_cast<const char*>(&mip.size.x), sizeof(mip.size.x));
                os.write(reinterpret_cast<const char*>(&mip.size.y), sizeof(mip.size.y));
                os.write(reinterpret_cast<const char*>(&mip.rcpSize.x), sizeof(mip.rcpSize.x));
                os.write(reinterpret_cast<const char*>(&mip.rcpSize.y), sizeof(mip.rcpSize.y));
                os.write(reinterpret_cast<const char*>(&mip.dataOffset), sizeof(mip.dataOffset));
                os.write(reinterpret_cast<const char*>(&mip.numElements), sizeof(mip.numElements));
                os.write(reinterpret_cast<const char*>(&mip.dataOffsetSAT), sizeof(mip.dataOffsetSAT));
            }
        }

        os.write(reinterpret_cast<const char*>(&m_tilingMode), sizeof(m_tilingMode));
        os.write(reinterpret_cast<const char*>(&m_textureFlags), sizeof(m_textureFlags));
        os.write(reinterpret_cast<const char*>(&m_alphaCutoff), sizeof(m_alphaCutoff));
        os.write(reinterpret_cast<const char*>(&m_textureFormat), sizeof(m_textureFormat));

        os.write(reinterpret_cast<const char*>(&m_dataSize), sizeof(m_dataSize));
        os.write(reinterpret_cast<const char*>(m_data), m_dataSize);

        os.write(reinterpret_cast<const char*>(&m_dataSATSize), sizeof(m_dataSATSize));
        if (m_dataSATSize != 0)
        {
            os.write(reinterpret_cast<const char*>(m_dataSAT), m_dataSATSize);
        }
    }

    template<class TMemoryStreamBuf>
    void TextureImpl::Deserialize(TMemoryStreamBuf& buffer, int inputDescVersion)
    {
        OMM_ASSERT(m_data == nullptr);
        OMM_ASSERT(m_dataSize == 0);
        OMM_ASSERT(m_dataSAT == nullptr);
        OMM_ASSERT(m_dataSATSize == 0);
        OMM_ASSERT(m_mips.size() == 0);

        std::istream os(&buffer);

        int numMips = 0;
        os.read(reinterpret_cast<char*>(&numMips), sizeof(numMips));

        if (numMips != 0)
        {
            m_mips.resize(numMips);
            for (auto& mip : m_mips)
            {
                os.read(reinterpret_cast<char*>(&mip.size.x), sizeof(mip.size.x));
                os.read(reinterpret_cast<char*>(&mip.size.y), sizeof(mip.size.y));
                os.read(reinterpret_cast<char*>(&mip.rcpSize.x), sizeof(mip.rcpSize.x));
                os.read(reinterpret_cast<char*>(&mip.rcpSize.y), sizeof(mip.rcpSize.y));
                os.read(reinterpret_cast<char*>(&mip.dataOffset), sizeof(mip.dataOffset));
                os.read(reinterpret_cast<char*>(&mip.numElements), sizeof(mip.numElements));
                os.read(reinterpret_cast<char*>(&mip.dataOffsetSAT), sizeof(mip.dataOffsetSAT));

                mip.sizeLog2.x = ctz(mip.size.x);
                mip.sizeLog2.y = ctz(mip.size.y);
                mip.sizef = (float2)mip.size;
                mip.sizeIsPow2 = isPow2(mip.size.x) && isPow2(mip.size.y);
            }
        }

        os.read(reinterpret_cast<char*>(&m_tilingMode), sizeof(m_tilingMode));

        if (inputDescVersion >= 3)
        {
            os.read(reinterpret_cast<char*>(&m_textureFlags), sizeof(m_textureFlags));
            os.read(reinterpret_cast<char*>(&m_alphaCutoff), sizeof(m_alphaCutoff));
        }
        else
        {
            if (m_tilingMode == TilingMode::MortonZ)
            {
                m_textureFlags = ommCpuTextureFlags_None;
            }
            else
            {
                m_textureFlags = ommCpuTextureFlags_DisableZOrder;
            }

            m_alphaCutoff = -1.f;
        }

        os.read(reinterpret_cast<char*>(&m_textureFormat), sizeof(m_textureFormat));

        os.read(reinterpret_cast<char*>(&m_dataSize), sizeof(m_dataSize));
        m_data = m_stdAllocator.allocate(m_dataSize, kAlignment);
        os.read(reinterpret_cast<char*>(m_data), m_dataSize);

        os.read(reinterpret_cast<char*>(&m_dataSATSize), sizeof(m_dataSATSize));
        if (m_dataSATSize != 0)
        {
            m_dataSAT = m_stdAllocator.allocate(m_dataSATSize, kAlignment);
            os.read(reinterpret_cast<char*>(m_dataSAT), m_dataSATSize);
        }
    }
}
