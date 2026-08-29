cmake_minimum_required(VERSION 3.20)
project(cmcore CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CMCORE_AVX2 "Enable AVX2/FMA fast paths" ON)

add_library(cmcore STATIC
  src/index.cpp
  src/store.cpp
)
target_include_directories(cmcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(cmcore PRIVATE -Wall -Wextra -Wpedantic)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if(CMCORE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    target_compile_options(cmcore PRIVATE -mavx2 -mfma)
  endif()
endif()

# --- core test suite (dependency-free) ---
enable_testing()
add_executable(cmcore_test tests/test_core.cpp)
target_link_libraries(cmcore_test PRIVATE cmcore)
add_test(NAME cmcore_test COMMAND cmcore_test)

# --- Python module (built by scikit-build-core; nanobind optional here) ---
find_package(Python COMPONENTS Interpreter Development QUIET)
find_package(nanobind CONFIG QUIET)
if(nanobind_FOUND AND Python_Development_FOUND)
  nanobind_add_module(_core python/bindings.cpp)
  target_link_libraries(_core PRIVATE cmcore)
  install(TARGETS _core LIBRARY DESTINATION contextmemory)
endif()