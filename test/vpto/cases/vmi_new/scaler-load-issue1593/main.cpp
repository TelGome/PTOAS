// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr) do { const aclError ret = (expr); if (ret != ACL_SUCCESS) { \
  std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)ret, __FILE__, __LINE__); \
  rc = 1; goto cleanup; } } while (0)

void LaunchScaler_load_issue1593_kernel(float *src, float *dst, void *stream);

int main() {
  constexpr size_t kElems = 600;
  const size_t bytes = kElems * sizeof(float);
  float *srcHost = nullptr, *dstHost = nullptr;
  float *srcDevice = nullptr, *dstDevice = nullptr;
  int rc = 0;
  bool aclInited = false, deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *env = std::getenv("ACL_DEVICE_ID")) deviceId = std::atoi(env);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void **)&srcHost, bytes));
  ACL_CHECK(aclrtMallocHost((void **)&dstHost, bytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dstDevice, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./v1.bin", bytes, srcHost, bytes);
  ReadFile("./v2.bin", bytes, dstHost, bytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, bytes, srcHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dstDevice, bytes, dstHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchScaler_load_issue1593_kernel(srcDevice, dstDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(dstHost, bytes, dstDevice, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", dstHost, bytes);

cleanup:
  aclrtFree(srcDevice); aclrtFree(dstDevice);
  aclrtFreeHost(srcHost); aclrtFreeHost(dstHost);
  if (stream) aclrtDestroyStream(stream);
  if (deviceSet) aclrtResetDevice(deviceId);
  if (aclInited) aclFinalize();
  return rc;
}
