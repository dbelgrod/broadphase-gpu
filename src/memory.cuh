#pragma once

#include <spdlog/spdlog.h>

namespace stq::gpu {

__device__ __host__ struct MemHandler {

  // pair in BP -> sizeof(int2) = 8 bytes
  // pair in NP -> 8*sizeof(double) = 64 bytes -> sizeof(CCDData)

  size_t MAX_OVERLAP_CUTOFF = 0;
  size_t MAX_OVERLAP_SIZE = 1e6;
  size_t MAX_UNIT_SIZE = 0;
  size_t MAX_QUERIES = 0;
  int realcount = 0;

  void setUnitSize(const size_t unit_size, const size_t constraint) {
    if (!MAX_UNIT_SIZE) {
      size_t free;
      size_t total;
      cudaMemGetInfo(&free, &total);
      size_t available_units = (free - constraint) * 0.9 / unit_size;
      spdlog::debug("unit options: available {:d} or overlap mulitplier {:d}",
                    available_units, 32 * MAX_QUERIES);
      MAX_UNIT_SIZE = std::min(available_units, 32 * MAX_QUERIES);
      return;
    }
  }

  void increaseUnitSize(const size_t constraint) {
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);
    spdlog::debug("Attempting to double w/ constraint {:d}", constraint);
    if ((0.9 * free - constraint) > 0) {
      MAX_UNIT_SIZE *= 2;
      spdlog::debug("Doubling unit_size to {:d}", MAX_UNIT_SIZE);
    }
  }

  void handleOverflow(const size_t constraint) {
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);
    if ((0.9 * free - constraint) > 0) {
      MAX_UNIT_SIZE *= 2;
      spdlog::debug("Doubling unit_size to {:d}", MAX_UNIT_SIZE);
    } else {
      MAX_QUERIES /= 2;
      spdlog::warn("Halving queries to {:d}", MAX_QUERIES);
    }
  }

  bool increaseOverlapSize(const float multiplier) {
    // overlaps (1) + vf_overlaps (1) + ee_overlaps(1) + queries(8) + unit_size
    // (1024) = 1035
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);

    int theta = 11; // 1035;
    size_t new_overlap_size = multiplier * MAX_OVERLAP_SIZE * sizeof(int2);
    if ((0.9 * free - theta * new_overlap_size) > 0) {
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
