#pragma once

#include <spdlog/spdlog.h>

namespace stq::gpu {

__device__ __host__ struct MemHandler {

  // pair in BP -> sizeof(int2) = 8 bytes
  // pair in NP -> 8*sizeof(double) = 64 bytes
  // we should aim for occupying 1/9 of the memory?

  size_t MAX_OVERLAP_CUTOFF = 0;
  size_t MAX_OVERLAP_SIZE = 1e6;
  int realcount = 0;

  bool increaseOverlapSize(const float multiplier) {
    // overlaps (1) + vf_overlaps (1) + ee_overlaps(1) + queries(8) + unit_size
    // (1024) = 1035
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);

    int theta = 11; // 1035;
    size_t new_overlap_size = multiplier * MAX_OVERLAP_SIZE * sizeof(int2);
    if (new_overlap_size >= total / theta) {
      spdlog::warn(
        "{:d} / {:d} bytes is insufficient to increase  {:.1f}x, shrinking cutoff",
        new_overlap_size, total, multiplier);
      return false;
    }
    MAX_OVERLAP_SIZE *= multiplier;
    spdlog::debug("Increasing size {:.1f}x to {:d}", multiplier,
                  MAX_OVERLAP_SIZE);
    return true;
  }

  void increaseOverlapCutoff(const float multiplier) {
    MAX_OVERLAP_CUTOFF *= multiplier;
    spdlog::debug("Increasing cutoff {:.1f}x to {:d}", multiplier,
                  MAX_OVERLAP_CUTOFF);
  }
};

extern MemHandler *memhandle;

// cudaMallocManaged(&memhandle, sizeof(MemHandler));

} // namespace stq::gpu
