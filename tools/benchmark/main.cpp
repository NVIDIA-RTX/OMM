/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <omm.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    struct Options
    {
        std::filesystem::path input = OMM_BENCHMARK_DEFAULT_INPUT;
        uint32_t iterations = 5;
        uint32_t warmup = 1;
        int threads = -1; // Preserve the serialized flags unless explicitly overridden.
        bool help = false;
    };

    void PrintUsage()
    {
        std::cout << "Usage: omm-cpu-benchmark [options]\n"
            "  --input PATH     Serialized bake input (default: bundled leaves.bin)\n"
            "  --iterations N   Measured iterations (default: 5)\n"
            "  --warmup N       Untimed warmup iterations (default: 1)\n"
            "  --threads on|off Override internal threading (default: serialized setting)\n"
            "  --help           Show this help\n";
    }

    uint32_t ParseCount(std::string_view text, bool allowZero)
    {
        uint32_t value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc() || end != text.data() + text.size() || (!allowZero && value == 0))
            throw std::runtime_error("Invalid iteration count: " + std::string(text));
        return value;
    }

    Options ParseOptions(int argc, char** argv)
    {
        Options options;
        for (int i = 1; i < argc; ++i)
        {
            const std::string_view argument = argv[i];
            if (argument == "--help" || argument == "-h")
            {
                options.help = true;
                return options;
            }
            if (argument != "--input" && argument != "--iterations" && argument != "--warmup" && argument != "--threads")
                throw std::runtime_error("Unknown option: " + std::string(argument));
            if (++i == argc)
                throw std::runtime_error("Missing value for " + std::string(argument));
            const std::string_view value = argv[i];
            if (argument == "--input")
                options.input = argv[i];
            else if (argument == "--iterations")
                options.iterations = ParseCount(value, false);
            else if (argument == "--warmup")
                options.warmup = ParseCount(value, true);
            else if (value == "on" || value == "off")
                options.threads = value == "on" ? 1 : 0;
            else
                throw std::runtime_error("--threads expects on or off");
        }
        return options;
    }

    void Check(omm::Result result, const char* operation)
    {
        if (result != omm::Result::SUCCESS)
            throw std::runtime_error(std::string(operation) + " failed (omm::Result " + std::to_string(static_cast<int>(result)) + ")");
    }

    struct Resources
    {
        omm::Baker baker = nullptr;
        omm::Cpu::DeserializedResult deserialized = nullptr;
        omm::Cpu::BakeResult result = nullptr;

        Resources() = default;
        Resources(const Resources&) = delete;
        Resources& operator=(const Resources&) = delete;

        ~Resources()
        {
            if (result)
                omm::Cpu::DestroyBakeResult(result);
            // Deserialization owns the textures and geometry referenced by the inputs.
            if (deserialized)
                omm::Cpu::DestroyDeserializedResult(deserialized);
            if (baker)
                omm::DestroyBaker(baker);
        }
    };

    int Run(const Options& options)
    {
        std::ifstream file(options.input, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("Cannot open input: " + options.input.string());
        const std::streamoff size = file.tellg();
        // The serialized format includes an eight-byte hash before its header.
        if (size < 8)
            throw std::runtime_error("Input is too short or unreadable: " + options.input.string());
        std::vector<uint8_t> data(static_cast<size_t>(size));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(data.data()), size))
            throw std::runtime_error("Cannot read input: " + options.input.string());

        Resources resources;
        omm::BakerCreationDesc creation;
        creation.type = omm::BakerType::CPU;
        creation.messageInterface.messageCallback = [](omm::MessageSeverity severity, const char* message, void*)
        {
            if (severity != omm::MessageSeverity::Info)
                std::cerr << "[omm-sdk] " << message << '\n';
        };
        Check(omm::CreateBaker(creation, &resources.baker), "CreateBaker");
        omm::Cpu::BlobDesc blob;
        blob.data = data.data();
        blob.size = data.size();
        Check(omm::Cpu::Deserialize(resources.baker, blob, &resources.deserialized), "Deserialize");
        const omm::Cpu::DeserializedDesc* deserialized = nullptr;
        Check(omm::Cpu::GetDeserializedDesc(resources.deserialized, &deserialized), "GetDeserializedDesc");
        if (deserialized->numInputDescs <= 0 || !deserialized->inputDescs)
            throw std::runtime_error("The blob contains no bake inputs");

        std::vector<omm::Cpu::BakeInputDesc> inputs(deserialized->inputDescs,
            deserialized->inputDescs + deserialized->numInputDescs);
        uint64_t triangleCount = 0;
        std::cout << "Input: " << options.input.string() << '\n';
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            auto& input = inputs[i];
            const auto threadFlag = static_cast<uint32_t>(omm::Cpu::BakeFlags::EnableInternalThreads);
            const auto flags = static_cast<uint32_t>(input.bakeFlags);
            if (options.threads >= 0)
                input.bakeFlags = static_cast<omm::Cpu::BakeFlags>(options.threads ? flags | threadFlag : flags & ~threadFlag);
            triangleCount += input.indexCount / 3;
            std::cout << "  Input " << i << ": " << input.indexCount / 3 << " triangles, max subdivision "
                << static_cast<unsigned>(input.maxSubdivisionLevel) << ", dynamic scale " << input.dynamicSubdivisionScale
                << ", internal threading " << ((static_cast<uint32_t>(input.bakeFlags) & threadFlag) ? "on" : "off") << '\n';
        }
        std::cout << "Warmup: " << options.warmup << ", measured iterations: " << options.iterations
            << " (bake calls only; loading and result cleanup excluded)\n" << std::flush;

        uint64_t arrayBytes = 0;
        uint64_t micromapCount = 0;
        const auto bake = [&]()
        {
            double milliseconds = 0;
            arrayBytes = 0;
            micromapCount = 0;
            for (const auto& input : inputs)
            {
                const auto start = std::chrono::steady_clock::now();
                const auto status = omm::Cpu::Bake(resources.baker, input, &resources.result);
                const auto end = std::chrono::steady_clock::now();
                Check(status, "Bake");
                milliseconds += std::chrono::duration<double, std::milli>(end - start).count();
                const omm::Cpu::BakeResultDesc* result = nullptr;
                Check(omm::Cpu::GetBakeResultDesc(resources.result, &result), "GetBakeResultDesc");
                arrayBytes += result->arrayDataSize;
                micromapCount += result->descArrayCount;
                Check(omm::Cpu::DestroyBakeResult(resources.result), "DestroyBakeResult");
                resources.result = nullptr;
            }
            return milliseconds;
        };

        for (uint32_t i = 0; i < options.warmup; ++i)
        {
            bake();
            std::cout << "Warmup " << i + 1 << '/' << options.warmup << " complete\n" << std::flush;
        }
        std::vector<double> times;
        times.reserve(options.iterations);
        std::cout << std::fixed << std::setprecision(3);
        for (uint32_t i = 0; i < options.iterations; ++i)
        {
            times.push_back(bake());
            std::cout << "Iteration " << i + 1 << '/' << options.iterations << ": " << times.back() << " ms\n" << std::flush;
        }
        const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        std::sort(times.begin(), times.end());
        const size_t middle = times.size() / 2;
        const double median = times.size() % 2 ? times[middle] : (times[middle - 1] + times[middle]) * 0.5;
        std::cout << "Bake time: min " << times.front() << " ms, median " << median << " ms, mean " << mean
            << " ms, max " << times.back() << " ms\n"
            << "Output: " << micromapCount << " micromaps, " << arrayBytes << " array bytes for "
            << triangleCount << " triangles\n";
        return 0;
    }
}

int main(int argc, char** argv)
{
    try
    {
        const auto options = ParseOptions(argc, argv);
        if (options.help)
        {
            PrintUsage();
            return 0;
        }
        return Run(options);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
