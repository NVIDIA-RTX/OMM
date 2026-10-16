/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>

#include "util/image.h"
#include "util/cpu_raster.h"
#include <omp.h>
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <vector>

namespace {

template <omm::RasterMode Mode, bool EnableBarycentrics>
void CompareSerialAndParallel(const omm::Triangle& triangle, int2 resolution, float2 offset)
{
	SCOPED_TRACE(::testing::Message() << "mode " << Mode << ", barycentrics " << EnableBarycentrics);
	const int2 min = int2(glm::floor(triangle.aabb_s * float2(resolution) + offset));
	const int2 max = int2(glm::ceil(triangle.aabb_e * float2(resolution) + offset));
	const int width = max.x - min.x;
	struct Sample {
		unsigned visits = 0;
		float3 barycentrics = float3(0.f);
	};
	std::vector<Sample> serial(width * (max.y - min.y)), parallel(serial.size());
	auto collect = [min, width](int2 index, const float3* barycentrics, void* context) {
		auto& samples = *static_cast<std::vector<Sample>*>(context);
		// Each scanline owns distinct slots, including cells outside the texture.
		auto& sample = samples.at((index.y - min.y) * width + index.x - min.x);
		++sample.visits;
		if (barycentrics)
			sample.barycentrics = *barycentrics;
	};
	if constexpr (EnableBarycentrics)
	{
		omm::RasterizeTriImpl<Mode, false, true>(triangle, resolution, offset, collect, &serial);
		omm::RasterizeTriImpl<Mode, true, true>(triangle, resolution, offset, collect, &parallel);
	}
	else
	{
		auto withoutBarycentrics = [&collect](int2 index, void* context) { collect(index, nullptr, context); };
		omm::RasterizeTriImpl<Mode, false, false>(triangle, resolution, offset, withoutBarycentrics, &serial);
		omm::RasterizeTriImpl<Mode, true, false>(triangle, resolution, offset, withoutBarycentrics, &parallel);
	}
	for (size_t i = 0; i < serial.size(); ++i)
	{
		ASSERT_EQ(serial[i].visits, parallel[i].visits) << "cell " << i;
		EXPECT_LE(serial[i].visits, 1u);
		if constexpr (EnableBarycentrics)
		{
			for (int component = 0; component < 3; ++component)
				EXPECT_FLOAT_EQ(serial[i].barycentrics[component], parallel[i].barycentrics[component]) << "cell " << i;
		}
	}
}

TEST(RasterTraversal, SerialMatchesParallel)
{
	const omm::Triangle triangles[] = {
		omm::Triangle({ 0.125f, 0.125f }, { 0.875f, 0.125f }, { 0.125f, 0.875f }),
		omm::Triangle({ -0.25f, -0.125f }, { 0.875f, 0.125f }, { -0.125f, 0.875f }),
		omm::Triangle({ -0.1f, -0.125f }, { 0.9f, 0.875f }, { 0.85f, 0.875f }),
		omm::Triangle({ 0.45f, 0.45f }, { 0.46f, 0.45f }, { 0.45f, 0.46f }),
	};
	for (size_t i = 0; i < std::size(triangles); ++i)
		for (bool reverse : { false, true })
			for (float shift : { 0.f, -0.5f })
			{
				SCOPED_TRACE(::testing::Message() << "triangle " << i << ", reverse " << reverse << ", offset " << shift);
				const auto& t = triangles[i];
				const omm::Triangle triangle = reverse ? omm::Triangle(t.p2, t.p1, t.p0) : t;
				const int2 resolution(16, 12);
				const float2 offset(shift);
				CompareSerialAndParallel<omm::Default, false>(triangle, resolution, offset);
				CompareSerialAndParallel<omm::Default, true>(triangle, resolution, offset);
				CompareSerialAndParallel<omm::OverConservative, false>(triangle, resolution, offset);
				CompareSerialAndParallel<omm::OverConservative, true>(triangle, resolution, offset);
				CompareSerialAndParallel<omm::UnderConservative, false>(triangle, resolution, offset);
				CompareSerialAndParallel<omm::UnderConservative, true>(triangle, resolution, offset);
			}
}

class RasterTest : public ::testing::TestWithParam<std::tuple<omm::Triangle, int2>> {
protected:
	omm::Triangle _triangle;
	int2 _size;
	void SetUp() override {
		_triangle = std::get<0>(GetParam());
		_size = std::get<1>(GetParam());

		EXPECT_EQ(_triangle.GetIsCCW(), true);
	}

	void TearDown() override {
	}

	const char* GetRasterModeName(omm::RasterMode mode) {
		if (omm::RasterMode::UnderConservative == mode)
		{
			return "UnderConservative";
		}
		if (omm::RasterMode::OverConservative == mode)
		{
			return "OverConservative";
		}
		else // if (omm::RasterMode::Default == mode)
		{
			return "Default";
		}
	}

	void Run(int2 size, omm::RasterMode mode, bool cw) {

		ImageRGB image(size, { 1, 128, 5 });

		omm::Triangle t = cw ? omm::Triangle(_triangle.p0, _triangle.p2, _triangle.p1) : _triangle;


		int one = 1;
		int checkerSize = 64;
		
		auto KernelCBFill = [&image, checkerSize](int2 idx, void* ) {
			if (idx.x >= image.GetSize().x)
				return;
			if (idx.y >= image.GetSize().y)
				return;
			if (idx.x < 0)
				return;
			if (idx.y < 0)
				return;

			float2 p = { checkerSize * float(idx.x / checkerSize + 0.5) / image.GetWidth() , checkerSize * float(idx.y / checkerSize + 0.5) / image.GetHeight() };
			uchar3 val;
			if ((idx.x / checkerSize) % 2 != (idx.y / checkerSize) % 2)
				val = { 0, 0, 0 };
			else
				val = { 64, 64, 64 };
			image.Store(idx, val);
		};

		struct Params {
			int checkerSize = 1;
			uchar3 fillColor = { 0,0 ,0 };
			bool fillWithBarycentrics = false;
		};

		auto TriangleFill = [&image, size](int2 idx, const float3* bc, void* ctx) {
			Params* p = (Params*)ctx;

			auto IsInRange = [&image](int2 idx)->bool {
				if (idx.x >= image.GetSize().x)
					return false;
				if (idx.y >= image.GetSize().y)
					return false;
				if (idx.x < 0)
					return false;
				if (idx.y < 0)
					return false;

				return true;
			};

			for (int y = 0; y < p->checkerSize; ++y) {
				for (int x = 0; x < p->checkerSize; ++x) {

					int2 dst = p->checkerSize * idx + int2{x, y};
					if (!IsInRange(dst))
						continue;


					uchar3 val = image.Load(dst);
					if (p->fillWithBarycentrics)
						val = uchar3(*bc * 200.f);
					else
						val.x += 64;
					image.Store(dst, val);
				}
			}
		};

		// "Fullscreen" pass.
		omm::RasterizeParallel(omm::Triangle({ 0.f, -1.f }, { 0.f, 1.f }, { 2.f, 1.f }), size, KernelCBFill);

		// "Triangle pass.
		if (omm::RasterMode::UnderConservative == mode)
		{
			{
				Params p;
				p.checkerSize = 1;
				p.fillWithBarycentrics = true;
				omm::RasterizeParallelBarycentrics(t, size, TriangleFill, &p);
			}

			{
				Params p;
				p.checkerSize = checkerSize;
				p.fillColor = uchar3(128, 0, 0);
				omm::RasterizeUnderConservativeBarycentrics(t, size / checkerSize, TriangleFill, &p);
			}
		}
		if (omm::RasterMode::OverConservative == mode)
		{
			{
				Params p;
				p.checkerSize = checkerSize;
				p.fillColor = uchar3(128, 0, 0);
				omm::RasterizeConservativeParallelBarycentrics(t, size / checkerSize, TriangleFill, &p);
			}

			{
				Params p;
				p.checkerSize = 1;
				p.fillWithBarycentrics = true;
				omm::RasterizeParallelBarycentrics(t, size, TriangleFill, &p);
			}
		}
		else if (omm::RasterMode::Default == mode)
		{
			{
				Params p;
				p.checkerSize = checkerSize;
				p.fillColor = uchar3(0, 0, 128);
				omm::RasterizeParallelBarycentrics(t, size / checkerSize, TriangleFill, &p);
			}

			{
				Params p;
				p.checkerSize = 1;
				p.fillWithBarycentrics = true;
				omm::RasterizeParallelBarycentrics(t, size, TriangleFill, &p);
			}
		}

		std::string name = ::testing::UnitTest::GetInstance()->current_test_suite()->name();
		std::replace(name.begin(), name.end(), '/', '_');



		std::string fileName = name + GetRasterModeName(mode) + (cw ? "_cw_" : "") + std::to_string(size.x) + "x" + std::to_string(size.y) + ".png";
		SaveImageToFile("RasterTestOutput", fileName.c_str(), image);
	}
};

TEST_P(RasterTest, Rasterize) {
	Run(_size, omm::RasterMode::Default, false);
}

TEST_P(RasterTest, RasterizeCW) {
	Run(_size, omm::RasterMode::Default, true);
}

TEST_P(RasterTest, RasterizeConservative) {
	Run(_size, omm::RasterMode::OverConservative, false);
}

TEST_P(RasterTest, RasterizeUnderConservative) {
	Run(_size, omm::RasterMode::UnderConservative, false);
}

TEST_P(RasterTest, RasterizeConservativeCW) {
	Run(_size, omm::RasterMode::OverConservative, true);
}

TEST_P(RasterTest, RasterizeSmall) {
	Run({ _size.x / 2, _size.y / 2, }, omm::RasterMode::Default, false);
}

TEST_P(RasterTest, RasterizeConservativeSmall) {
	Run({ _size.x / 2, _size.y / 2, }, omm::RasterMode::OverConservative, false);
}

TEST_P(RasterTest, RasterizeLarge) {
	Run({ _size.x * 2, _size.y * 2, }, omm::RasterMode::Default, false);
}

TEST_P(RasterTest, RasterizeConservativeLarge) {
	Run({ _size.x * 2, _size.y * 2, }, omm::RasterMode::OverConservative, false);
}

TEST_P(RasterTest, RasterizeConservativeSuperLarge) {
	Run({ _size.x * 4, _size.y * 4, }, omm::RasterMode::OverConservative, false);
}

INSTANTIATE_TEST_SUITE_P(
	RasterContained,
	RasterTest,
	::testing::Values(
		std::make_tuple<omm::Triangle, int2>(omm::Triangle({0.2f, 0.2f}, {0.7f, 0.5f},{0.3f, 0.8f}), {1024, 1024})
	));

INSTANTIATE_TEST_SUITE_P(
	RasterSubPixel,
	RasterTest,
	::testing::Values(
		std::make_tuple<omm::Triangle, int2>(omm::Triangle({ 0.2f, 0.2f }, { 0.21f, 0.21f }, { 0.2f, 0.21f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterSubPixelMaxCoverage,
	RasterTest,
	::testing::Values(
		std::make_tuple<omm::Triangle, int2>(omm::Triangle({ 0.2f, 0.2f }, { 0.25f, 0.24f }, { 0.2f, 0.25f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterPartiallyCovered,
	RasterTest,
	::testing::Values(
		std::make_tuple<omm::Triangle, int2>(omm::Triangle({ -0.1f, -0.1f }, { 1.1f, -0.1f }, { -0.1f, 1.1f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterPartiallyCovered2,
	RasterTest,
	::testing::Values(
		std::make_tuple<omm::Triangle, int2>(omm::Triangle({ -0.2f, 0.2f }, { 0.7f, 0.5f }, { -0.3f, 0.8f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterFullyCovered,
	RasterTest,
	::testing::Values(
		std::make_tuple<omm::Triangle, int2>(omm::Triangle({ -0.1f, -1.1f }, { 2.1f, 1.1f }, { -0.1f, 1.1f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterBorked,
	RasterTest,
	::testing::Values(
		std::make_tuple<omm::Triangle, int2>(omm::Triangle({ 0.609000027f, 0.332400024f }, { 0.332400024f, 0.402599990f }, { 0.402599990f, 0.332400024f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterBorked2,
	RasterTest,
	::testing::Values(
		std::make_tuple<omm::Triangle, int2>(omm::Triangle({ 0.609000027f, 0.332400024f },  { 0.332400024f, 0.402599990f }, { 0.402599990f, 0.332400024f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterBorked3,
	RasterTest,
	::testing::Values(
		std::make_tuple<omm::Triangle, int2>(omm::Triangle({ 0.809000027f, 0.332400024f }, { 0.332400024f, 0.502599990f }, { 0.402599990f, 0.332400024f }), { 1024, 1024 })
	));

}  // namespace
