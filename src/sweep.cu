#include <cuda/pipeline>

// #include <stq/gpu/aabb.cuh>
#include <stq/gpu/queue.cuh>
#include <stq/gpu/sweep.cuh>

#include <spdlog/spdlog.h>

namespace stq::gpu {

__global__ void build_index(Aabb *boxes, int N, int *index) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N)
    return;

  index[tid] = tid;
}

__global__ void retrieve_collision_pairs(const Aabb *const boxes, int *count,
                                         int2 *overlaps, int N, int guess,
                                         int nbox, int start, int end) {
  extern __shared__ Aabb s_objects[];

  // int gid = blockIdx.x * blockDim.x + threadIdx.x;
  // if (gid == 0)
  //     printf("index[0]-> %i\n", index[0]);

  int tid = start + threadIdx.x + blockIdx.x * blockDim.x;
  int ltid = threadIdx.x;

  if (tid >= N || tid >= end)
    return;

  // #pragma unroll
  // for (int i=0; i < nbox; i++)
  // {
  //     int t = tid + i*blockDim.x;
  //     int l = i*blockDim.x + ltid;
  //     s_objects[l]= boxes[t];
  // }
  s_objects[ltid] = boxes[tid];

  __syncthreads();

  int t = tid + 0 * blockDim.x;
  int l = 0 * blockDim.x + ltid;
  // tid = tid + 1*blockDim.x;
  // ltid = 1*blockDim.x + ltid;

  int ntid = t + 1;
  int nltid = l + 1;

  if (ntid >= N)
    return;

  const Aabb &a = s_objects[l];
  Aabb b = nltid < blockDim.x ? s_objects[nltid] : boxes[ntid];
  int i = 0;
  while (a.max.x >= b.min.x) // boxes can touch and collide
  {
    // printf("res %i %i\n", tid, ntid);
    i++;
    if (does_collide(a, b) && !covertex(a.vertexIds, b.vertexIds)
        // if (tid % 100 == 0
    )
      // add_overlap(0, 0, count, overlaps, guess);
      add_overlap(a.id, b.id, count, overlaps, guess);

    ntid++;
    nltid++;
    if (ntid >= N)
      return;
    b = nltid < blockDim.x ? s_objects[nltid] : boxes[ntid];
  }
  if (tid == 0)
    printf("final count for box 0: %i\n", i);
}

__device__ void consider_pair(const int &xid, const int &yid, int *count,
                              int2 *out, int guess) {
  int i = atomicAdd(count, 1);

  if (i < guess) {
    out[i] = make_int2(xid, yid);
  }
}

// __global__ void create_sortedmin(Aabb * boxes, Scalar3 * sortedmin,
// int N)
// __global__ void average(Aabb * boxes, Scalar3 * sm, MiniBox * mini,
// int N, Scalar3 * mean)
__global__ void calc_mean(Aabb *boxes, Scalar3 *mean, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N)
    return;

  // add to mean

  Scalar3 mx = __fdividef((boxes[tid].min + boxes[tid].max), 2 * N);
  atomicAdd(&mean[0].x, mx.x);
  atomicAdd(&mean[0].y, mx.y);
  atomicAdd(&mean[0].z, mx.z);

  // atomicAdd(&mean[0].x, __fdividef((boxes[tid].min.x + boxes[tid].max.x),
  // 2*N)); atomicAdd(&mean[0].y, __fdividef((boxes[tid].min.y +
  // boxes[tid].max.y), 2*N)); atomicAdd(&mean[0].z,
  // __fdividef((boxes[tid].min.z + boxes[tid].max.z), 2*N));
}

__global__ void calc_variance(Aabb *boxes, Scalar3 *var, int N, Scalar3 *mean) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N)
    return;

  Scalar3 fx = __powf(abs(boxes[tid].min - mean[0]), 2.0) +
               __powf(abs(boxes[tid].max - mean[0]), 2.0);
  // if (tid == 0) spdlog::trace("{:.6f} {:.6f} {:.6f}", fx.x, fx.y, fx.z);
  atomicAdd(&var[0].x, fx.x);
  atomicAdd(&var[0].y, fx.y);
  atomicAdd(&var[0].z, fx.z);
}

__global__ void create_ds(Aabb *boxes, Scalar2 *sortedmin, MiniBox *mini, int N,
                          Dimension axis) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N)
    return;
  Scalar *min;
  Scalar *max;

  if (axis == x) {
    sortedmin[tid] = make_Scalar2(boxes[tid].min.x, boxes[tid].max.x);
    min = (Scalar[2]){boxes[tid].min.y, boxes[tid].min.z};
    max = (Scalar[2]){boxes[tid].max.y, boxes[tid].max.z};
  } else if (axis == y) {

    sortedmin[tid] = make_Scalar2(boxes[tid].min.y, boxes[tid].max.y);
    min = (Scalar[2]){boxes[tid].min.x, boxes[tid].min.z};
    max = (Scalar[2]){boxes[tid].max.x, boxes[tid].max.z};
  } else {
    sortedmin[tid] = make_Scalar2(boxes[tid].min.z, boxes[tid].max.z);
    min = (Scalar[2]){boxes[tid].min.x, boxes[tid].min.y};
    max = (Scalar[2]){boxes[tid].max.x, boxes[tid].max.y};
  }

  // sm[tid] = SortedMin(boxes[tid].min.x, boxes[tid].max.x, tid,
  // boxes[tid].vertexIds);

  // Scalar min[2] = {boxes[tid].min.y, boxes[tid].min.z};
  // Scalar max[2] = {boxes[tid].max.y, boxes[tid].max.z};

  mini[tid] = MiniBox(tid, min, max, boxes[tid].vertexIds);
}

// __global__ void build_checker(Scalar3 * sortedmin, int2 * out, int
// N, int * count, int guess)
__global__ void build_checker(Scalar3 *sm, int2 *out, int N, int *count,
                              int guess) {
  // Scalar3 x -> min, y -> max, z-> boxid
  extern __shared__ Scalar3 s_sortedmin[];
  // __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
  int nbox = 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N)
    return;

  int ltid = threadIdx.x;

  // init(&barrier, 1);

  // cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  // pipe.producer_acquire();
  // cuda::memcpy_async(s_sortedmin + ltid, sm + tid, sizeof(*sm), pipe);
  // pipe.producer_commit();
  // pipe.consumer_wait();

  // pipeline_producer_commit(pipe, barrier);
  // barrier.arrive_and_wait();
  // pipe.consumer_release();

  for (int i = 0; i < nbox; i++)
    s_sortedmin[i * blockDim.x + ltid] = sm[tid + i * blockDim.x];

  __syncthreads();

  // for (int i=0; i<nbox;i++)
  // {
  int ntid = tid + 1;
  int nltid = ltid + 1;
  // int ntid = tid+i*blockDim.x + 1;
  // int nltid = i*blockDim.x + ltid + 1;

  if (ntid >= N)
    return;

  const Scalar3 &a = s_sortedmin[ltid];
  Scalar3 b = nltid < nbox * blockDim.x ? s_sortedmin[nltid] : sm[ntid];
  // const Scalar3& a = sortedmin[tid];
  // Scalar3 b = sortedmin[ntid];

  while (a.y >= b.x) // curr max > following min
  {
    // if(!covertex(a.vertexIds, b.vertexIds))
    consider_pair(int(a.z), int(b.z), count, out, guess);

    ntid++;
    nltid++;
    if (ntid >= N)
      return;
    b = nltid < nbox * blockDim.x ? s_sortedmin[nltid] : sm[ntid];
    // b = sortedmin[ntid];
  }
  // }
}

__global__ void twostage_queue(Scalar2 *sm, const MiniBox *const mini,
                               int2 *overlaps, int N, int *count, int guess,
                               int *start, int *end, const int MAX_OVERLAP_SIZE) {
  __shared__ Queue queue;
  queue.heap_size = HEAP_SIZE;
  queue.start = 0;
  queue.end = 0;

  int tid = threadIdx.x + blockIdx.x * blockDim.x + *start;
  if (tid >= N || tid + 1 >= N)
    return;

  // if (count >= MAX_OVERLAP_SIZE && blockIdx.x > maxBlockId )
  //   return;

  Scalar2 a = sm[tid];
  Scalar2 b = sm[tid + 1];

  if (a.y >= b.x) {
    int2 val = make_int2(tid, tid + 1);
    queue.push(val);
  }
  __syncthreads();
  queue.nbr_per_loop = queue.end - queue.start;

  while (queue.nbr_per_loop > 0) {
    if (threadIdx.x >= queue.nbr_per_loop)
      return;
    int2 res = queue.pop();
    MiniBox ax = mini[res.x];
    MiniBox bx = mini[res.y];

    if (does_collide(ax, bx) && is_valid_pair(ax.vertexIds, bx.vertexIds) &&
        !covertex(ax.vertexIds, bx.vertexIds)) {
      add_overlap(ax.id, bx.id, count, overlaps, guess);
    }

    if (res.y + 1 >= N)
      return;
    a = sm[res.x];
    b = sm[res.y + 1];
    if (a.y >= b.x) {
      res.y += 1;
      queue.push(res);
    }
    __syncthreads();
    queue.nbr_per_loop = queue.end - queue.start;
    queue.nbr_per_loop = queue.nbr_per_loop < 0
                           ? queue.end + HEAP_SIZE - queue.start
                           : queue.nbr_per_loop;
  }
}

} // namespace stq::gpu