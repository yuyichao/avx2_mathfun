cmake_minimum_required(VERSION 2.8)
project(avx2_mathfun)

include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES avx2_mathfun.cpp)
add_library(avx2_mathfun SHARED ${SOURCE_FILES})
set_target_properties(avx2_mathfun PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -ffast-math")
install(TARGETS avx2_mathfun DESTINATION ${CMAKE_INSTALL_LIBDIR})
