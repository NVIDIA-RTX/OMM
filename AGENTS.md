# Repository Guidelines

## Project Structure & Module Organization

This is a CMake-based C++20 SDK. Core code lives in `libraries/omm-lib`, with public headers in `libraries/omm-lib/include` and implementation in `libraries/omm-lib/src`. The NVRHI GPU integration layer is in `libraries/omm-gpu-nvrhi`, and the viewer application is in `tools/viewer`. Project tests are under `support/tests`; test inputs and example content live in `assets/tests` and `assets/omm_example_data`. Documentation is in `docs`, while third-party dependencies are submodules under `external`. Treat `build/` and `Bin/` as generated output, not source.

## Build, Test, and Development Commands

Initialize dependencies before configuring:

```powershell
git submodule update --init --recursive
```

Windows MSVC configuration and build:

```powershell
cmake -S . -B build -A x64 -DOMM_VIEWER_INSTALL=OFF
cmake --build build --config Release
```

Linux/Ninja configuration and build:

```sh
cmake -S . -B build -G Ninja -DOMM_VIEWER_INSTALL=OFF
cmake --build build --config Release
```

Run tests after a Release build. CI skips GPU tests on Windows with:

```powershell
.\build\bin\Release\tests.exe --gtest_filter=-*GPU*
```

On single-config Linux builds, run `./build/bin/tests`. Install artifacts with `cmake --install build --config Release`.

## Coding Style & Naming Conventions

Follow the surrounding C++ style; there is no repository-level `.clang-format`. Keep code C++20-compatible, prefer helper types in `libraries/omm-lib/src`, and avoid broad refactors in third-party or generated files. Public API additions belong in `omm.h`/`omm.hpp`; implementation details should stay in `src`. Tests use `test_*.cpp` files and GoogleTest names like `TEST(Baker, CreateDestroy)` or `TEST_F(GpuTest, Pipeline)`. Existing code commonly uses PascalCase for API functions and types, camelCase for locals, and `_member` for private members.

## Testing Guidelines

Use GoogleTest via the `tests` target in `support/tests`. Add focused regression tests for baker behavior, indexing, serialization, shader pipeline setup, or viewer-facing data changes. If image output is needed for diagnosis, configure with `-DOMM_TEST_ENABLE_IMAGE_DUMP=ON`; otherwise keep it off to avoid noisy artifacts. There is no stated coverage threshold, so document the test command and platform in the PR.

## Commit & Pull Request Guidelines

Recent commits use short, direct subjects such as `Fix validation layer error`, `Adding tests for u16 index buffer`, or `Updating dependencies (#83)`. Keep subjects action-oriented and specific. Pull requests should summarize the change, list affected platforms or APIs, link issues when relevant, and include the exact build/test commands run. Include screenshots or captures for viewer UI changes.

## Agent-Specific Notes

Do not modify `external/` dependency contents unless the task is explicitly about vendored code. Prefer small, reviewable SDK changes and keep generated build products out of commits.
