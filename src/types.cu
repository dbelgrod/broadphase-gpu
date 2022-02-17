#include <stq/gpu/types.cuh>

#include <cuda_fp16.h>

namespace stq::gpu {

// __host__ __device__ half3 make_half3(__half x, __half y, __half z) {
//   half3 t;
//   t.x = x;
//   t.y = y;
//   t.z = z;
//   return t;
// }

// __host__ __device__ half3 make_half3(float x, float y, float z) {
//   half3 t;
//   t.x = __float2half(x);
//   t.y = __float2half(y);
//   t.z = __float2half(z);
//   return t;
// }

#ifdef CCD_USE_DOUBLE

__host__ __device__ Scalar3 make_Scalar3(const Scalar a, const Scalar b,
                                         const Scalar c) {
  return make_double3(a, b, c);
}

__host__ __device__ Scalar2 make_Scalar2(const Scalar a, const Scalar b) {
  return make_double2(a, b);
}

#else

__host__ __device__ Scalar3 make_Scalar3(const Scalar a, const Scalar b,
                                         const Scalar c) {
  return make_float3(a, b, c);
}

__host__ __device__ Scalar2 make_Scalar2(const Scalar a, const Scalar b) {
  return make_float2(a, b);
}

#endif

} // namespace stq::gpu

__device__ stq::gpu::Scalar3 operator+(const stq::gpu::Scalar3 &a,
                                       const stq::gpu::Scalar3 &b) {
  return stq::gpu::make_Scalar3(__fadd_rz(a.x, b.x), __fadd_rz(a.y, b.y),
                                __fadd_rz(a.z, b.z));
}

__device__ stq::gpu::Scalar3 operator-(const stq::gpu::Scalar3 &a,
                                       const stq::gpu::Scalar3 &b) {
  return stq::gpu::make_Scalar3(__fsub_rz(a.x, b.x), __fsub_rz(a.y, b.y),
                                __fsub_rz(a.z, b.z));
}

__device__ stq::gpu::Scalar3 __fdividef(const stq::gpu::Scalar3 &a,
                                        const stq::gpu::Scalar b) {
  return stq::gpu::make_Scalar3(__fdividef(a.x, b), __fdividef(a.y, b),
                                __fdividef(a.z, b));
}

__device__ stq::gpu::Scalar3 __powf(const stq::gpu::Scalar3 &a,
                                    const stq::gpu::Scalar b) {
  return stq::gpu::make_Scalar3(__powf(a.x, b), __powf(a.y, b), __powf(a.z, b));
}

__device__ stq::gpu::Scalar3 abs(const stq::gpu::Scalar3 &a) {
  return stq::gpu::make_Scalar3(__habs(a.x), __habs(a.y), __habs(a.z));
}
