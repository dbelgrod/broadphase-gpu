#pragma once

#include <spdlog/spdlog.h>
#include <cuda/semaphore>

namespace stq::gpu {

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    spdlog::error("GPUassert: {} {} {:d}", cudaGetErrorString(code), file,
                  line);
    if (abort)
      exit(code);
  }
}

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

  size_t MAX_OVERLAP_CUTOFF = 0;
  size_t MAX_OVERLAP_SIZE = 1e6;
  size_t MAX_UNIT_SIZE = 0;
  size_t MAX_QUERIES = 0;
  int realcount = 0;
  int limitGB = 0;

  size_t __getAllocatable() {
    size_t free;
    size_t total;
    gpuErrchk(cudaMemGetInfo(&free, &total));

    size_t used = total - free;

    size_t defaultAllocatable = 0.95 * free;
    size_t tmp = static_cast<size_t>(limitGB) * 1073741824;
    size_t userAllocatable = tmp - used;

    spdlog::info("Can allocate {:.2f}% of memory",
                 static_cast<float>(userAllocatable) / total * 100);

    return std::min(defaultAllocatable, userAllocatable);
    ;
  }

  void setUnitSize(const size_t constraint) {
    size_t allocatable = __getAllocatable();
    size_t available_units = (allocatable - constraint) / sizeof(MP_unit);
    spdlog::trace("unit options: available {:d} or overlap mulitplier {:d}",
                  available_units, 32 * MAX_QUERIES);
    MAX_UNIT_SIZE = std::min(available_units, 32 * MAX_QUERIES);
    spdlog::trace("Set unit_size to {:.2f}% of allocatable mem",
                  static_cast<float>(MAX_UNIT_SIZE) * sizeof(MP_unit) /
                    allocatable * 100);
    return;
  }

  void increaseUnitSize(const size_t constraint) {
    size_t allocatable = __getAllocatable();
    spdlog::trace("Attempting to double w/ constraint {:d}", constraint);
    if ((allocatable - constraint) > 0) {
      MAX_UNIT_SIZE *= 2;
      spdlog::trace("Doubling unit_size to {:.2f}% of allocatable mem",
                    static_cast<float>(MAX_UNIT_SIZE) * sizeof(MP_unit) /
                      allocatable * 100);
    }
  }

  void handleOverflow(const size_t constraint) {
    size_t allocatable = __getAllocatable();
    if ((allocatable - constraint) > 0) {
      MAX_UNIT_SIZE *= 2;
      spdlog::trace("Doubling unit_size to {:.2f}% of allocatable mem",
                    static_cast<float>(MAX_UNIT_SIZE) * sizeof(MP_unit) /
                      allocatable * 100);
    } else {
      MAX_QUERIES /= 2;
      spdlog::warn("Halving queries to {:d}", MAX_QUERIES);
    }
  }

  void handleBroadPhaseOverflow(int desired_count) {
    size_t allocatable = __getAllocatable();

    size_t largest_overlap_size =
      (allocatable - sizeof(CCDConfig)) /
      (sizeof(CCDData) + sizeof(MP_unit) + 3 * sizeof(int2));
    MAX_OVERLAP_SIZE =
      std::min(largest_overlap_size, static_cast<size_t>(desired_count));
    spdlog::info(
      "Setting MAX_OVERLAP_SIZE to {:.2f}% ({:d}) of allocatable memory",
      static_cast<float>(MAX_OVERLAP_SIZE) * sizeof(int2) / allocatable * 100,
      MAX_OVERLAP_SIZE);

    if (MAX_OVERLAP_SIZE == largest_overlap_size) {
      MAX_OVERLAP_CUTOFF *= 0.5;
      spdlog::warn(
        "Insufficient memory to increase overlap size, shrinking cutoff 0.5x to {:d}",
        MAX_OVERLAP_CUTOFF);
    }
  }
};

extern MemHandler *memhandle;

// cudaMallocManaged(&memhandle, sizeof(MemHandler));

} // namespace stq::gpu
