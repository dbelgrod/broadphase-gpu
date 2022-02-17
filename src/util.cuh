#pragma once

#include <spdlog/spdlog.h>

namespace stq::gpu {

#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      spdlog::error("Fatal error: {:s} ({:s} at {:s}:{:d})", msg,              \
                    cudaGetErrorString(__err), __FILE__, __LINE__);            \
      spdlog::error("FAILED - ABORTING");                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void setup(int devId, int &smemSize, int &threads, int &nbox);

} // namespace stq::gpu