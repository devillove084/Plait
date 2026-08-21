include_guard(GLOBAL)

set(_plait_use_system_dependencies FALSE)
if(PLAIT_DEPENDENCY_MODE STREQUAL "AUTO" OR
   PLAIT_DEPENDENCY_MODE STREQUAL "SYSTEM")
  set(_plait_use_system_dependencies TRUE)
endif()

if(_plait_use_system_dependencies)
  find_package(absl CONFIG QUIET)
  if(BUILD_TESTING AND PLAIT_BUILD_TESTS)
    find_package(GTest CONFIG QUIET)
  endif()
  if(PLAIT_ENABLE_YALANTINGLIBS)
    find_package(yalantinglibs CONFIG QUIET)
  endif()
  if(PLAIT_ENABLE_OPENCL)
    find_package(OpenCL QUIET)
  endif()
endif()

set(_plait_fetch_abseil FALSE)
set(_plait_fetch_googletest FALSE)
set(_plait_fetch_yalantinglibs FALSE)
set(_plait_fetch_opencl FALSE)

if(NOT TARGET absl::hash)
  set(_plait_fetch_abseil TRUE)
endif()
if(BUILD_TESTING AND PLAIT_BUILD_TESTS AND NOT TARGET GTest::gtest_main)
  set(_plait_fetch_googletest TRUE)
endif()
if(PLAIT_ENABLE_YALANTINGLIBS AND
   NOT TARGET yalantinglibs::yalantinglibs)
  set(_plait_fetch_yalantinglibs TRUE)
endif()
if(PLAIT_ENABLE_OPENCL AND NOT TARGET OpenCL::OpenCL)
  set(_plait_fetch_opencl TRUE)
endif()

if(PLAIT_DEPENDENCY_MODE STREQUAL "SYSTEM")
  set(_plait_missing_dependencies)
  if(_plait_fetch_abseil)
    list(APPEND _plait_missing_dependencies absl)
  endif()
  if(_plait_fetch_googletest)
    list(APPEND _plait_missing_dependencies GTest)
  endif()
  if(_plait_fetch_yalantinglibs)
    list(APPEND _plait_missing_dependencies yalantinglibs)
  endif()
  if(_plait_fetch_opencl)
    list(APPEND _plait_missing_dependencies OpenCL)
  endif()
  if(_plait_missing_dependencies)
    list(JOIN _plait_missing_dependencies ", " _plait_missing_text)
    message(FATAL_ERROR
      "SYSTEM dependency mode could not find: ${_plait_missing_text}"
    )
  endif()
endif()

if(_plait_fetch_abseil OR _plait_fetch_googletest OR
   _plait_fetch_yalantinglibs OR _plait_fetch_opencl)
  include(FetchContent)
  set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

  # Declare every source dependency before populating any of them. This keeps
  # the top-level project authoritative if dependencies share subdependencies.
  if(_plait_fetch_abseil)
    FetchContent_Declare(abseil
      URL
        https://codeload.github.com/abseil/abseil-cpp/tar.gz/5650e9cf76d3be4318d5fa3af38ee483ddfd5e4a
      URL_HASH
        SHA256=bac569e31429f85d7bdd9f90e4767442faf9c8fa1def08a66a7d5d78e0fba204
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM
      EXCLUDE_FROM_ALL
    )
  endif()

  if(_plait_fetch_googletest)
    FetchContent_Declare(googletest
      URL
        https://codeload.github.com/google/googletest/tar.gz/52eb8108c5bdec04579160ae17225d66034bd723
      URL_HASH
        SHA256=745c55415660044610f7fcd3af7a6420d5de16a7dbb9ebfe2e131275676232be
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM
      EXCLUDE_FROM_ALL
    )
  endif()

  if(_plait_fetch_yalantinglibs)
    FetchContent_Declare(yalantinglibs
      URL
        https://codeload.github.com/alibaba/yalantinglibs/tar.gz/refs/tags/0.6.1
      URL_HASH
        SHA256=2ef2089a49a08f764c558e9caf46e1d37697b111e04a48e5c5156f57f3afff24
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM
      EXCLUDE_FROM_ALL
    )
  endif()

  if(_plait_fetch_opencl)
    # The loader's CMake project defines both OpenCL::OpenCL and
    # OpenCL::Headers. Populate the headers without adding their CMake project
    # to avoid defining OpenCL::Headers twice.
    FetchContent_Declare(opencl_headers
      URL
        https://codeload.github.com/KhronosGroup/OpenCL-Headers/tar.gz/refs/tags/v2026.05.29
      URL_HASH
        SHA256=d9e6c48357de5002da11ce45de600e0c3ffe6ab4f628a3b9fe2b38603161658a
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SOURCE_SUBDIR _plait_do_not_add_subdirectory
      SYSTEM
      EXCLUDE_FROM_ALL
    )
    FetchContent_Declare(opencl_icd_loader
      URL
        https://codeload.github.com/KhronosGroup/OpenCL-ICD-Loader/tar.gz/refs/tags/v2026.05.29
      URL_HASH
        SHA256=48fd0c5181db7cd046f4f731d5955694892e10998d49d09ee0d997e7e04fd939
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM
      EXCLUDE_FROM_ALL
    )
  endif()

  if(_plait_fetch_abseil)
    set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(abseil)
  endif()

  if(_plait_fetch_googletest)
    set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  if(_plait_fetch_yalantinglibs)
    set(INSTALL_STANDALONE OFF CACHE BOOL "" FORCE)
    set(INSTALL_THIRDPARTY OFF CACHE BOOL "" FORCE)
    set(YLT_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    set(YLT_ENABLE_DL OFF CACHE BOOL "" FORCE)
    set(YLT_ENABLE_FILE_IO_URING OFF CACHE BOOL "" FORCE)
    set(YLT_ENABLE_IBV OFF CACHE BOOL "" FORCE)
    set(YLT_ENABLE_IO_URING OFF CACHE BOOL "" FORCE)
    set(YLT_ENABLE_NTLS OFF CACHE BOOL "" FORCE)
    set(YLT_ENABLE_PMR OFF CACHE BOOL "" FORCE)
    set(YLT_ENABLE_SSL OFF CACHE BOOL "" FORCE)
    set(YLT_ENABLE_STRUCT_PACK_OPTIMIZE OFF CACHE BOOL "" FORCE)
    set(YLT_ENABLE_STRUCT_PACK_UNPORTABLE_TYPE OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(yalantinglibs)
  endif()

  if(_plait_fetch_opencl)
    set(OPENCL_HEADERS_BUILD_CXX_TESTS OFF CACHE BOOL "" FORCE)
    set(OPENCL_HEADERS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(opencl_headers)

    set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(OPENCL_ICD_LOADER_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(OPENCL_ICD_LOADER_HEADERS_DIR "${opencl_headers_SOURCE_DIR}"
      CACHE PATH "" FORCE
    )
    FetchContent_MakeAvailable(opencl_icd_loader)
  endif()
endif()

if(NOT TARGET absl::hash)
  message(FATAL_ERROR "Abseil did not provide the required hash target")
endif()
if(BUILD_TESTING AND PLAIT_BUILD_TESTS AND NOT TARGET GTest::gtest_main)
  message(FATAL_ERROR "GoogleTest did not provide GTest::gtest_main")
endif()
if(PLAIT_ENABLE_YALANTINGLIBS AND
   NOT TARGET yalantinglibs::yalantinglibs)
  message(FATAL_ERROR "YalantingLibs did not provide its CMake target")
endif()
if(PLAIT_ENABLE_OPENCL AND NOT TARGET OpenCL::OpenCL)
  message(FATAL_ERROR "OpenCL did not provide OpenCL::OpenCL")
endif()
