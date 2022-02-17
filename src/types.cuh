#pragma once

#include <algorithm>

#include <stdint.h>
#include <tbb/info.h>

namespace stq::gpu {

static const int CPU_THREADS = std::min(tbb::info::default_concurrency(), 64);

// __host__ __device__ struct half3 {
//   __half x;
//   __half y;
//   __half z;
// };

// __host__ __device__ half3 make_half3(__half x, __half y, __half z);

// __host__ __device__ half3 make_half3(float x, float y, float z);

typedef enum { x, y, z } Dimension;

typedef unsigned long long int ull;

#ifdef CCD_USE_DOUBLE
typedef double Scalar;
typedef double2 Scalar2;
typedef double3 Scalar3;
__host__ __device__ Scalar2 make_Scalar2(const Scalar a, const Scalar b);
__host__ __device__ Scalar3 make_Scalar3(const Scalar a, const Scalar b,
                                         const Scalar c);
#else
typedef float Scalar;
typedef float2 Scalar2;
typedef float3 Scalar3;
__host__ __device__ Scalar2 make_Scalar2(const Scalar a, const Scalar b);
__host__ __device__ Scalar3 make_Scalar3(const Scalar a, const Scalar b,
                                         const Scalar c);
#endif

} // namespace stq::gpu

__device__ stq::gpu::Scalar3 operator+(const stq::gpu::Scalar3 &a,
                                       const stq::gpu::Scalar3 &b);
__device__ stq::gpu::Scalar3 operator-(const stq::gpu::Scalar3 &a,
                                       const stq::gpu::Scalar3 &b);
__device__ stq::gpu::Scalar3 __fdividef(const stq::gpu::Scalar3 &a,
                                        const stq::gpu::Scalar b);
__device__ stq::gpu::Scalar3 __powf(const stq::gpu::Scalar3 &a,
                                    const stq::gpu::Scalar b);
__device__ stq::gpu::Scalar3 abs(const stq::gpu::Scalar3 &a);
