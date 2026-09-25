# fsd_core — the C++ hot-path core for the FSD stack (Phase 1 migration).
#
# Targets:
#   fsd_core   static library: types + safety monitor + control + occupancy
#   fsd_smoke  synthetic end-to-end scenario executable (also a ctest)
#   fsd_cpp    pybind11 extension module — only with -DFSD_PYBIND=ON
#
#   cmake -B build -S cpp
#   cmake --build build --config Release
#   ctest --test-dir build
cmake_minimum_required(VERSION 3.20)
project(fsd_core VERSION 0.1.0 LANGUAGES CXX)

option(FSD_PYBIND "Build the fsd_cpp Python extension module (needs pybind11)" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(fsd_core STATIC
    src/safety.cpp
    src/control.cpp
    src/occupancy.cpp
    bindings/pybind_module.cpp   # no-op TU here; real module below
)
target_include_directories(fsd_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(fsd_core PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(fsd_core PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(fsd_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(fsd_smoke src/smoke_main.cpp)
target_link_libraries(fsd_smoke PRIVATE fsd_core)

enable_testing()
add_test(NAME fsd_smoke COMMAND fsd_smoke)

# ------------------------------------------------------------- fsd_cpp module
if(FSD_PYBIND)
  find_package(Python3 COMPONENTS Interpreter Development QUIET)

  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND AND Python3_Interpreter_FOUND)
    # Fall back to `python -m pybind11 --cmakedir` (covers pip installs).
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
      OUTPUT_VARIABLE _pybind11_cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _pybind11_probe)
    if(_pybind11_probe EQUAL 0 AND EXISTS "${_pybind11_cmakedir}")
      find_package(pybind11 CONFIG REQUIRED
                   PATHS "${_pybind11_cmakedir}" NO_DEFAULT_PATH)
    endif()
  endif()
  if(NOT pybind11_FOUND)
    message(FATAL_ERROR
      "FSD_PYBIND=ON but pybind11 was not found. "
      "`pip install pybind11` or pass -Dpybind11_DIR=<...>/share/cmake/pybind11.")
  endif()

  pybind11_add_module(fsd_cpp bindings/pybind_module.cpp)
  target_link_libraries(fsd_cpp PRIVATE fsd_core)
  target_compile_definitions(fsd_cpp PRIVATE FSD_WITH_PYBIND)
  if(MSVC)
    target_compile_options(fsd_cpp PRIVATE /utf-8)
  endif()
endif()
