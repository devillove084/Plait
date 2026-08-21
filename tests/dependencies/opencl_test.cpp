#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <gtest/gtest.h>

TEST(DependencyContract, OpenClLoaderCanEnumerate) {
  cl_uint platform_count = 0;
  const cl_int status = clGetPlatformIDs(0, nullptr, &platform_count);

  EXPECT_TRUE(status == CL_SUCCESS || status == CL_PLATFORM_NOT_FOUND_KHR);
}
