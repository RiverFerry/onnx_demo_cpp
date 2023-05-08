cmake_minimum_required(VERSION 3.7)

set(TARGET onnxCXX)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_VERBOSE_MAKEFILE on)
set(build_DIR ${CMAKE_SOURCE_DIR}/build)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
string(REPLACE "-Wl,--exclude-libs,libgcc_real.a" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# Release or Debug
set (CMAKE_BUILD_TYPE "Debug")
if(${CMAKE_BUILD_TYPE} MATCHES "Release")
    message(STATUS "Release版本")
    set(BuildType "Release")
    SET(CMAKE_C_FLAGS "$ENV{CFLAGS} -DNODEBUG -O3 -Wall")
    SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -DNODEBUG -O3 -Wall")
    MESSAGE(STATUS "CXXFLAGS: " ${CMAKE_CXX_FLAGS})
    MESSAGE(STATUS "CFLAGS: " ${CMAKE_C_FLAGS})
else()
    set(BuildType "Debug")
    message(STATUS "Debug版本")
    SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -Wall -O0 -g")
    #   SET(CMAKE_C_FILAGS "-O0 -g")
    SET(CMAKE_C_FLAGS "$ENV{CFLAGS} -O0 -g")
    MESSAGE(STATUS "CXXFLAGS: " ${CMAKE_CXX_FLAGS})
    MESSAGE(STATUS "CFLAGS: " ${CMAKE_C_FILAGS})
endif()

link_directories(${CMAKE_SOURCE_DIR}/lib)
link_libraries(onnxruntime)
add_executable(${TARGET} main.cpp)

include_directories(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
)
