cmake_minimum_required(VERSION 3.28)

project(
  Plait
  VERSION 0.1.0
  DESCRIPTION "Rankless runtime for asynchronous distributed tensors"
  LANGUAGES C CXX
)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Plait has one language dialect. Dependencies and every Plait target are
# configured as C++26; there is intentionally no C++23 compatibility mode.
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PLAIT_BUILD_TESTS "Build Plait tests" ON)
option(PLAIT_ENABLE_INSTALL "Generate the experimental Plait SDK package" OFF)
option(PLAIT_ENABLE_CLANG_TIDY "Run clang-tidy on Plait targets" ON)
option(PLAIT_ENABLE_OPENCL "Build the OpenCL adapter foundation" ON)
option(PLAIT_ENABLE_YALANTINGLIBS "Build the struct_pack wire-codec foundation" ON)
option(PLAIT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
set(PLAIT_DEPENDENCY_MODE "AUTO" CACHE STRING
  "Dependency source: AUTO, SYSTEM, or FETCH"
)
set_property(CACHE PLAIT_DEPENDENCY_MODE PROPERTY STRINGS AUTO SYSTEM FETCH)
set(PLAIT_SANITIZER "address" CACHE STRING
  "Runtime sanitizer: none, address, or thread"
)
set_property(CACHE PLAIT_SANITIZER PROPERTY STRINGS none address thread)

string(TOUPPER "${PLAIT_DEPENDENCY_MODE}" PLAIT_DEPENDENCY_MODE)
if(NOT PLAIT_DEPENDENCY_MODE MATCHES "^(AUTO|SYSTEM|FETCH)$")
  message(FATAL_ERROR "PLAIT_DEPENDENCY_MODE must be AUTO, SYSTEM, or FETCH")
endif()

if(NOT PLAIT_SANITIZER MATCHES "^(none|address|thread)$")
  message(FATAL_ERROR "PLAIT_SANITIZER must be none, address, or thread")
endif()

if(NOT PLAIT_SANITIZER STREQUAL "none" AND
   NOT CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang|GNU)$")
  message(FATAL_ERROR "PLAIT_SANITIZER requires Clang, AppleClang, or GCC")
endif()

if(PLAIT_ENABLE_CLANG_TIDY)
  find_program(PLAIT_CLANG_TIDY_PROGRAM NAMES clang-tidy REQUIRED)
endif()
find_program(PLAIT_CLANG_FORMAT_PROGRAM NAMES clang-format REQUIRED)

include(cmake/Dependencies.cmake)

set(PLAIT_CORE_SOURCES
  src/core/placement.cpp
  src/core/tensor.cpp
)

add_library(plait_core ${PLAIT_CORE_SOURCES})
add_library(Plait::Core ALIAS plait_core)

target_compile_features(plait_core PUBLIC cxx_std_26)
set_target_properties(plait_core PROPERTIES EXPORT_NAME Core)

target_include_directories(plait_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(plait_core
  PRIVATE
    absl::hash
)

function(plait_apply_build_constraints target)
  set_target_properties(${target} PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
  )

  if(PLAIT_ENABLE_CLANG_TIDY)
    set_target_properties(${target} PROPERTIES
      CXX_CLANG_TIDY
        "${PLAIT_CLANG_TIDY_PROGRAM};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy"
    )
  endif()

  if(MSVC)
    target_compile_options(${target} PRIVATE
      /W4
      /permissive-
      $<$<BOOL:${PLAIT_WARNINGS_AS_ERRORS}>:/WX>
    )
  else()
    target_compile_options(${target} PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wsign-conversion
      -Wshadow
      -Wformat=2
      -Wundef
      -Wdouble-promotion
      -Wimplicit-fallthrough
      $<$<BOOL:${PLAIT_WARNINGS_AS_ERRORS}>:-Werror>
    )
  endif()

  if(PLAIT_SANITIZER STREQUAL "address")
    target_compile_options(${target} PRIVATE
      -fsanitize=address,undefined
      -fno-omit-frame-pointer
      -fno-optimize-sibling-calls
      -fno-sanitize-recover=all
    )
    target_link_options(${target} PRIVATE
      -fsanitize=address,undefined
      -fno-sanitize-recover=all
    )
  elseif(PLAIT_SANITIZER STREQUAL "thread")
    target_compile_options(${target} PRIVATE
      -fsanitize=thread
      -fno-omit-frame-pointer
      -fno-optimize-sibling-calls
    )
    target_link_options(${target} PRIVATE -fsanitize=thread)
  endif()
endfunction()

plait_apply_build_constraints(plait_core)

set(PLAIT_TEST_SOURCES tests/core/core_test.cpp)
if(PLAIT_ENABLE_YALANTINGLIBS)
  list(APPEND PLAIT_TEST_SOURCES tests/dependencies/struct_pack_test.cpp)
endif()
if(PLAIT_ENABLE_OPENCL)
  list(APPEND PLAIT_TEST_SOURCES tests/dependencies/opencl_test.cpp)
endif()

file(GLOB_RECURSE PLAIT_FORMAT_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

set(PLAIT_TIDY_FILES ${PLAIT_CORE_SOURCES} ${PLAIT_TEST_SOURCES})
list(TRANSFORM PLAIT_TIDY_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

add_custom_target(format
  COMMAND ${PLAIT_CLANG_FORMAT_PROGRAM} -i ${PLAIT_FORMAT_FILES}
  COMMENT "Formatting Plait sources with Google clang-format"
  VERBATIM
)
add_custom_target(format-check
  COMMAND ${PLAIT_CLANG_FORMAT_PROGRAM} --dry-run --Werror ${PLAIT_FORMAT_FILES}
  COMMENT "Checking Plait source formatting"
  VERBATIM
)

if(PLAIT_ENABLE_CLANG_TIDY)
  add_custom_target(tidy
    COMMAND ${PLAIT_CLANG_TIDY_PROGRAM}
      --config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy
      -p ${CMAKE_CURRENT_BINARY_DIR}
      ${PLAIT_TIDY_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running clang-tidy across Plait sources"
    VERBATIM
  )
endif()

if(BUILD_TESTING AND PLAIT_BUILD_TESTS)
  add_executable(plait_core_tests ${PLAIT_TEST_SOURCES})
  target_link_libraries(plait_core_tests PRIVATE Plait::Core GTest::gtest_main)

  if(PLAIT_ENABLE_YALANTINGLIBS)
    target_link_libraries(plait_core_tests PRIVATE yalantinglibs::yalantinglibs)
  endif()
  if(PLAIT_ENABLE_OPENCL)
    target_link_libraries(plait_core_tests PRIVATE OpenCL::OpenCL)
    target_compile_definitions(plait_core_tests PRIVATE CL_TARGET_OPENCL_VERSION=300)
  endif()

  plait_apply_build_constraints(plait_core_tests)
  add_test(NAME plait.core COMMAND plait_core_tests)

  if(PLAIT_SANITIZER STREQUAL "address")
    set_tests_properties(plait.core PROPERTIES ENVIRONMENT
      "ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1:strict_string_checks=1:halt_on_error=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1"
    )
  elseif(PLAIT_SANITIZER STREQUAL "thread")
    set_tests_properties(plait.core PROPERTIES ENVIRONMENT
      "TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1"
    )
  endif()
endif()

if(PLAIT_ENABLE_INSTALL)
  if(NOT PLAIT_DEPENDENCY_MODE STREQUAL "SYSTEM")
    message(FATAL_ERROR
      "PLAIT_ENABLE_INSTALL currently requires PLAIT_DEPENDENCY_MODE=SYSTEM "
      "so exported targets reference installed dependency packages"
    )
  endif()

  install(
    TARGETS plait_core
    EXPORT PlaitTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  set(PLAIT_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/Plait)
  configure_package_config_file(
    cmake/PlaitConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PlaitConfig.cmake
    INSTALL_DESTINATION ${PLAIT_INSTALL_CMAKEDIR}
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/PlaitConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  install(
    EXPORT PlaitTargets
    NAMESPACE Plait::
    DESTINATION ${PLAIT_INSTALL_CMAKEDIR}
  )
  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/PlaitConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/PlaitConfigVersion.cmake
    DESTINATION ${PLAIT_INSTALL_CMAKEDIR}
  )
endif()
