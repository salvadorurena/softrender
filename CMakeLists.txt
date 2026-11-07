cmake_minimum_required(VERSION 3.20)
project(softrender LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

# --- Core renderer: zero third-party dependencies ---------------------------
add_library(sr STATIC
  src/mesh.cpp
  src/texture.cpp
  src/png.cpp
  src/scene.cpp
)
target_include_directories(sr PUBLIC include)
target_link_libraries(sr PUBLIC Threads::Threads)
target_compile_definitions(sr PUBLIC SR_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
if(MSVC)
  target_compile_options(sr PUBLIC /W4)
else()
  target_compile_options(sr PUBLIC -Wall -Wextra -Wpedantic)
endif()

# --- Headless renderer: writes PNGs, runs benchmarks ------------------------
add_executable(softrender apps/render_cli.cpp)
target_link_libraries(softrender PRIVATE sr)

# --- Asset generator: procedurally builds the bundled OBJ models ------------
add_executable(mkassets tools/mkassets.cpp)
target_link_libraries(mkassets PRIVATE sr)

# --- Interactive viewer (optional, needs SDL3) ------------------------------
find_package(SDL3 CONFIG QUIET)
if(SDL3_FOUND)
  add_executable(viewer apps/viewer.cpp)
  target_link_libraries(viewer PRIVATE sr SDL3::SDL3)
  message(STATUS "SDL3 found: building interactive viewer")
else()
  message(STATUS "SDL3 not found: skipping interactive viewer (headless renderer still builds)")
endif()

# --- Tests ------------------------------------------------------------------
enable_testing()
add_executable(sr_tests tests/test_main.cpp)
target_link_libraries(sr_tests PRIVATE sr)
target_compile_definitions(sr_tests PRIVATE SR_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME sr_tests COMMAND sr_tests)
# End-to-end: render the default scene to a PNG, and run a short benchmark.
add_test(NAME render_smoke COMMAND softrender --size 640x360 --ssaa 2 --wireframe
         --out ${CMAKE_CURRENT_BINARY_DIR}/smoke.png)
add_test(NAME bench_smoke COMMAND softrender --bench 5 --size 320x180)
