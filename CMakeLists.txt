cmake_minimum_required(VERSION 3.3)

project(Matchem CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(kokkos)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Woverloaded-virtual -Wnon-virtual-dtor -Wreturn-type -Wswitch-enum -Wsign-compare -Wfloat-equal -Wparentheses -Wreorder -Wshadow -Wold-style-cast -Wextra -D_GLIBCXX_DEBUG")

# Cmake options
set(MEM_DEBUG FALSE CACHE BOOL "Enable memory sanitizing. Requires Gcc/clang. May require setting LD_PRELOAD to <path>/libasan.so (default False)")
set(GDB_ATTACH FALSE CACHE BOOL "Allow user to attach gdb when assertions are tripped (default False)")

if (MEM_DEBUG)
  set(CMAKE_CXX_FLAGS "-fsanitize=address -fno-omit-frame-pointer ${CMAKE_CXX_FLAGS}")
endif()

file(GLOB LIB_SRCS "*.cpp")
add_library(matchemlib ${LIB_SRCS})
target_link_libraries(matchemlib PUBLIC kokkos ${PYTHON_LIBRARIES})
target_include_directories(matchemlib SYSTEM PUBLIC ${PYTHON_INCLUDE_DIRS})
if (GDB_ATTACH)
  target_compile_definitions(matchemlib PUBLIC MATCHEMLIB_ATTACH)
endif()

add_executable(matchem main.C)
target_link_libraries(matchem matchemlib)

enable_testing()
include(CTest)

add_subdirectory(tests)
