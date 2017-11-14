cmake_minimum_required(VERSION 2.8)
project(avx2_mathfun)

include(GNUInstallDirs)

set(MAJOR_VERSION 0)
set(MINOR_VERSION 1)
set(PATCH_VERSION 0)

set(INSTALL_PKGCONFIG_DIR "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig"
  CACHE PATH "Installation directory for pkgconfig (.pc) files")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# -ffast-math produces 20% faster code on skylake with GCC 7.2.0
set(A2M_COMPILE_FLAGS "-mavx2 -mfma -ffast-math")

set(SOURCE_FILES avx2_mathfun.cpp)
add_library(avx2_mathfun SHARED ${SOURCE_FILES})
set_target_properties(avx2_mathfun PROPERTIES
  COMPILE_FLAGS "${A2M_COMPILE_FLAGS}"
  VERSION ${MAJOR_VERSION}.${MINOR_VERSION}
  SOVERSION ${MAJOR_VERSION})
configure_file(avx2_mathfun.pc.in avx2_mathfun.pc @ONLY)
install(TARGETS avx2_mathfun DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES avx2_mathfun.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_BINARY_DIR}/avx2_mathfun.pc
  DESTINATION ${INSTALL_PKGCONFIG_DIR})

if(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
  include(InstallRequiredSystemLibraries)

  set(CPACK_SET_DESTDIR "on")
  set(CPACK_PACKAGING_INSTALL_PREFIX "/tmp")
  set(CPACK_GENERATOR "DEB")

  set(CPACK_PACKAGE_DESCRIPTION "AVX2 math library")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AVX2 math library")
  set(CPACK_PACKAGE_VENDOR "Harvard")
  set(CPACK_PACKAGE_CONTACT "Yichao Yu")
  set(CPACK_PACKAGE_VERSION_MAJOR "${MAJOR_VERSION}")
  set(CPACK_PACKAGE_VERSION_MINOR "${MINOR_VERSION}")
  set(CPACK_PACKAGE_VERSION_PATCH "${PATCH_VERSION}")
  set(CPACK_PACKAGE_FILE_NAME
    "${CMAKE_PROJECT_NAME}_${MAJOR_VERSION}.${MINOR_VERSION}.${CPACK_PACKAGE_VERSION_PATCH}")
  set(CPACK_SOURCE_PACKAGE_FILE_NAME
    "${CMAKE_PROJECT_NAME}_${MAJOR_VERSION}.${MINOR_VERSION}.${CPACK_PACKAGE_VERSION_PATCH}")

  set(CPACK_DEBIAN_PACKAGE_DEPENDS "")

  set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
  set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
  set(CPACK_DEBIAN_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})

  set(CPACK_COMPONENTS_ALL Libraries ApplicationData)
  include(CPack)
endif()
