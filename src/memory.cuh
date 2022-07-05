#pragma once

#include <spdlog/spdlog.h>
#include <cuda/semaphore>

namespace stq::gpu {

class CCDConfig {
public:
  double co_domain_tolerance; // tolerance of the co-domain
  unsigned int mp_start;
  unsigned int mp_end;
  int mp_remaining;
  long unit_size;
  double toi;
  cuda::binary_semaphore<cuda::thread_scope_device> mutex;
  bool use_ms;
  bool allow_zero_toi;
  int max_iter;
  int overflow_flag;
};

class Singleinterval {
public:
  double first;
  double second;
};

class MP_unit {
public:
  Singleinterval itv[3];
  int query_id;
};

class CCDData {
public:
  double v0s[3];
  double v1s[3];
  double v2s[3];
  double v3s[3];
  double v0e[3];
  double v1e[3];
  double v2e[3];
  double v3e[3];
  double ms;     // minimum separation
  double err[3]; // error bound of each query, calculated from each scene
  double tol[3]; // domain tolerance to help decide which dimension to split
#ifdef CCD_TOI_PER_QUERY
  double toi;
  int aid;
  int bid;
#endif
  int nbr_checks = 0;
};

__device__ __host__ struct MemHandler {

  // pair in BP -> sizeof(int2) = 8 bytes
  // pair in NP -> 8*sizeof(double) = 64 bytes -> sizeof(CCDData)

  size_t MAX_OVERLAP_CUTOFF = 0;
  size_t MAX_OVERLAP_SIZE = 1e6;
  size_t MAX_UNIT_SIZE = 0;
  size_t MAX_QUERIES = 0;
  int realcount = 0;

  void setUnitSize(const size_t constraint) {
    // if (!MAX_UNIT_SIZE) {
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);
    size_t available_units = (0.95 * free - constraint) / sizeof(MP_unit);
    spdlog::debug("unit options: available {:d} or overlap mulitplier {:d}",
                  available_units, 32 * MAX_QUERIES);
    MAX_UNIT_SIZE = std::min(available_units, 32 * MAX_QUERIES);
    spdlog::debug("Set unit_size to {:.2f}% of free mem",
                  static_cast<float>(MAX_UNIT_SIZE) * sizeof(MP_unit) / free *
                    100);
    return;
    // }
  }

  void increaseUnitSize(const size_t constraint) {
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);
    spdlog::debug("Attempting to double w/ constraint {:d}", constraint);
    if ((0.95 * free - constraint) > 0) {
      MAX_UNIT_SIZE *= 2;
      spdlog::debug("Doubling unit_size to {:.2f}% of free mem",
                    static_cast<float>(MAX_UNIT_SIZE) * sizeof(MP_unit) / free *
                      100);
    }
  }

  void handleOverflow(const size_t constraint) {
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);
    if ((0.95 * free - constraint) > 0) {
      MAX_UNIT_SIZE *= 2;
      spdlog::debug("Doubling unit_size to {:.2f}% of free mem",
                    static_cast<float>(MAX_UNIT_SIZE) * sizeof(MP_unit) / free *
                      100);
    } else {
      MAX_QUERIES /= 2;
      spdlog::warn("Halving queries to {:d}", MAX_QUERIES);
    }
  }

  bool increaseOverlapSize(int desired_count) {
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);

    // // size_t new_ov /rlap_size = 2 * memhandle->MAX_OVERLAP_SIZE;
    // size_t contraint = sizeof(CCDConfig) + sizeof(CCDData) * new_overlap_size
    // +
    //                    sizeof(MP_unit) * new_overlap_size +
    //                    sizeof(int2) * new_overlap_size * 3;

    size_t largest_overlap_size =
      (.95 * free - sizeof(CCDConfig)) /
      (sizeof(CCDData) + sizeof(MP_unit) + 3 * sizeof(int2));
    MAX_OVERLAP_SIZE =
      std::min(largest_overlap_size, static_cast<size_t>(desired_count));
    spdlog::debug("Setting MAX_OVERLAP_SIZE to {:.2f}% ({:d}) of free memory",
                  static_cast<float>(MAX_OVERLAP_SIZE) * sizeof(int2) / free *
                    100,
                  MAX_OVERLAP_SIZE);

    if (MAX_OVERLAP_SIZE == largest_overlap_size) {
      spdlog::warn(
        "Insufficient memory for 2x MAX_OVERLAP_SIZE, shrinking cutoff");
      return false;
    }
    // MAX_OVERLAP_SIZE = (0.95 * free - constraint) / sizeof(int2);
    spdlog::debug("Setting MAX_OVERLAP_SIZE to {:.2f}% ({:d}) of free memory",
                  static_cast<float>(MAX_OVERLAP_SIZE) * sizeof(int2) / free *
                    100,
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
