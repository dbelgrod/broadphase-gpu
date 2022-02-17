
#include <gpubf/aabb.cuh>
#include <gpubf/util.cuh>

#include <spdlog/spdlog.h>

using namespace stq::gpu;

void setup(int devId, int &smemSize, int &threads, int &nbox) {
  // Host code
  // int maxbytes = 98304; // 96 KB
  // cudaFuncSetAttribute(get_collision_pairs,
  // cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

  int maxSmem;
  cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlock, devId);
  spdlog::trace("Max shared Memory per Block: {:d} B", maxSmem);

  int maxThreads;
  cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, devId);
  spdlog::trace("Max threads per Block: {:d} thrds", maxThreads);

  nbox = nbox ? nbox : std::max((int)(maxSmem / sizeof(Aabb)) / maxThreads, 1);
  spdlog::trace("Boxes per Thread: {:d}", nbox);

  // divide threads by an arbitrary number as long as its reasonable >64
  if (!threads) {
    cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor,
                           devId);

    spdlog::trace("Max threads per Multiprocessor: {:d} thrds", threads);
  }
  smemSize = nbox * threads * sizeof(Aabb);

  while (smemSize > maxSmem || threads > maxThreads) {
    threads--;
    smemSize = nbox * threads * sizeof(Aabb);
  }
  spdlog::trace("Actual threads per Block: {:d} thrds", threads);
  spdlog::trace("Shared mem alloc: {:d} B", smemSize);

  // int warpSize;
  // cudaDeviceGetAttribute(&warpSize,
  //     cudaDevAttrWarpSize, devId);
  // spdlog::trace("Warp Size: {:d}", warpSize);

  // bank conflict avoid
  // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  // cudaSharedMemConfig bankSize;
  // cudaDeviceGetSharedMemConfig(&bankSize);
  // spdlog::trace("Bank size: {:d}", bankSize );

  return;
}