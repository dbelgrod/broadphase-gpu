#include <cuda/pipeline>

// #include <gpubf/aabb.cuh>
#include <gpubf/queue.cuh>
#include <gpubf/sweep.cuh>

using namespace ccdgpu;

__global__ void build_index(Aabb *boxes, int N, int *index) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N)
    return;

  index[tid] = tid;
}

// __global__ void sort_sorting_axis(SweepMarker* axis)
// {

// }

__global__ void print_sort_axis(Aabb *axis, int C) {
  // for (uint i = 0; i < C; i++)
  //     printf("id: %i, x: %.6f\n", axis[i].id, axis[i].min.x);
}

__global__ void print_overlap_start(int2 *overlaps) {
  printf("overlap[0].x %d\n", overlaps[0].x);
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

  while (a.max.x >= b.min.x) // boxes can touch and collide
  {
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
}

__device__ void consider_pair(const int &xid, const int &yid, int *count,
                              int2 *out, int guess) {
  int i = atomicAdd(count, 1);

  if (i < guess) {
    out[i] = make_int2(xid, yid);
  }
}

__device__ ccdgpu::Scalar3 operator+(const ccdgpu::Scalar3 &a,
                                     const ccdgpu::Scalar3 &b) {

  return make_Scalar3(__fadd_rz(a.x, b.x), __fadd_rz(a.y, b.y),
                      __fadd_rz(a.z, b.z));
}

__device__ ccdgpu::Scalar3 __fdividef(const ccdgpu::Scalar3 &a,
                                      const Scalar &b) {

  return make_Scalar3(__fdividef(a.x, b), __fdividef(a.y, b),
                      __fdividef(a.z, b));
}

// __global__ void create_sortedmin(Aabb * boxes, ccdgpu::Scalar3 * sortedmin,
// int N)
// __global__ void average(Aabb * boxes, ccdgpu::Scalar3 * sm, MiniBox * mini,
// int N, ccdgpu::Scalar3 * mean)
__global__ void calc_mean(Aabb *boxes, ccdgpu::Scalar3 *mean, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N)
    return;

  // add to mean

  ccdgpu::Scalar3 mx = __fdividef((boxes[tid].min + boxes[tid].max), 2 * N);
  atomicAdd(&mean[0].x, mx.x);
  atomicAdd(&mean[0].y, mx.y);
  atomicAdd(&mean[0].z, mx.z);

  // atomicAdd(&mean[0].x, __fdividef((boxes[tid].min.x + boxes[tid].max.x),
  // 2*N)); atomicAdd(&mean[0].y, __fdividef((boxes[tid].min.y +
  // boxes[tid].max.y), 2*N)); atomicAdd(&mean[0].z,
  // __fdividef((boxes[tid].min.z + boxes[tid].max.z), 2*N));
}

// #include <math.h>

__device__ ccdgpu::Scalar3 operator-(const ccdgpu::Scalar3 &a,
                                     const ccdgpu::Scalar3 &b) {

  return make_Scalar3(__fsub_rz(a.x, b.x), __fsub_rz(a.y, b.y),
                      __fsub_rz(a.z, b.z));
}

__device__ ccdgpu::Scalar3 __powf(const ccdgpu::Scalar3 &a, const Scalar &b) {
  return make_Scalar3(__powf(a.x, b), __powf(a.y, b), __powf(a.z, b));
}

__device__ ccdgpu::Scalar3 abs(const ccdgpu::Scalar3 &a) {
  return make_Scalar3(__habs(a.x), __habs(a.y), __habs(a.z));
}

__global__ void calc_variance(Aabb *boxes, ccdgpu::Scalar3 *var, int N,
                              ccdgpu::Scalar3 *mean) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N)
    return;

  ccdgpu::Scalar3 fx = __powf(abs(boxes[tid].min - mean[0]), 2.0) +
                       __powf(abs(boxes[tid].max - mean[0]), 2.0);
  // if (tid == 0) printf("%.6f %.6f %.6f\n", fx.x, fx.y, fx.z);
  atomicAdd(&var[0].x, fx.x);
  atomicAdd(&var[0].y, fx.y);
  atomicAdd(&var[0].z, fx.z);
}

__global__ void create_ds(Aabb *boxes, ccdgpu::Scalar3 *sortedmin,
                          MiniBox *mini, int N, Dimension axis) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N)
    return;
  Scalar *min;
  Scalar *max;

  if (axis == x) {
    sortedmin[tid] =
        make_Scalar3(boxes[tid].min.x, boxes[tid].max.x, Scalar(tid));
    min = (Scalar[2]){boxes[tid].min.y, boxes[tid].min.z};
    max = (Scalar[2]){boxes[tid].max.y, boxes[tid].max.z};
  } else if (axis == y) {

    sortedmin[tid] =
        make_Scalar3(boxes[tid].min.y, boxes[tid].max.y, Scalar(tid));
    min = (Scalar[2]){boxes[tid].min.x, boxes[tid].min.z};
    max = (Scalar[2]){boxes[tid].max.x, boxes[tid].max.z};
  } else {
    sortedmin[tid] =
        make_Scalar3(boxes[tid].min.z, boxes[tid].max.z, Scalar(tid));
    min = (Scalar[2]){boxes[tid].min.x, boxes[tid].min.y};
    max = (Scalar[2]){boxes[tid].max.x, boxes[tid].max.y};
  }

  // sm[tid] = SortedMin(boxes[tid].min.x, boxes[tid].max.x, tid,
  // boxes[tid].vertexIds);

  // Scalar min[2] = {boxes[tid].min.y, boxes[tid].min.z};
  // Scalar max[2] = {boxes[tid].max.y, boxes[tid].max.z};

  mini[tid] = MiniBox(min, max, boxes[tid].vertexIds);
}

// __global__ void build_checker(ccdgpu::Scalar3 * sortedmin, int2 * out, int N,
// int * count, int guess)
__global__ void build_checker(ccdgpu::Scalar3 *sm, int2 *out, int N, int *count,
                              int guess) {
  // ccdgpu::Scalar3 x -> min, y -> max, z-> boxid
  extern __shared__ ccdgpu::Scalar3 s_sortedmin[];
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

  const ccdgpu::Scalar3 &a = s_sortedmin[ltid];
  ccdgpu::Scalar3 b = nltid < nbox * blockDim.x ? s_sortedmin[nltid] : sm[ntid];
  // const ccdgpu::Scalar3& a = sortedmin[tid];
  // ccdgpu::Scalar3 b = sortedmin[ntid];

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

// __global__ void retrieve_collision_pairs2(const Aabb* const boxes, int *
// count, int2 * inpairs, int2 * overlaps, int N, int guess)
__global__ void retrieve_collision_pairs2(const MiniBox *const mini, int *count,
                                          int2 *inpairs, int2 *overlaps, int N,
                                          int guess) {
  extern __shared__ MiniBox s_mini[];

  int nbox = 1;

  int tid = threadIdx.x + nbox * blockIdx.x * blockDim.x;
  int ltid = threadIdx.x;

  if (tid >= N)
    return;

  for (int i = 0; i < nbox; i++) {
    int aid = inpairs[tid + i * blockDim.x].x;
    int bid = inpairs[tid + i * blockDim.x].y;

    // s_mini[2*(ltid+i*blockDim.x)] = mini[aid];
    // s_mini[2*(ltid+i*blockDim.x)+1] = mini[bid];

    // const MiniBox& a = s_mini[2*(ltid+i*blockDim.x)];
    // const MiniBox& b = s_mini[2*(ltid+i*blockDim.x)+1];

    const MiniBox &a = mini[aid];
    const MiniBox &b = mini[bid];

    if (does_collide(a, b) && !covertex(a.vertexIds, b.vertexIds)) {
      add_overlap(aid, bid, count, overlaps, guess);
    }
  }
}

__global__ void twostage_queue(ccdgpu::Scalar3 *sm, const MiniBox *const mini,
                               int2 *overlaps, int N, int *count, int guess,
                               int start, int end) {
  if (threadIdx.x + blockIdx.x * blockDim.x == 0)
    printf("sizeof(ccdgpu::Scalar3) %i\n", sizeof(ccdgpu::Scalar3));
  // extern __shared__ ccdgpu::Scalar3 s_sortedmin[];
  // __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2>
  // pss;
  __shared__ Queue queue;
  queue.capacity = 4000;
  queue.heap_size = 4000;
  __syncthreads();
  for (int i = threadIdx.x; i < queue.capacity; i += blockDim.x) {
    queue.lock[i].release();
    queue.harr[i].x = -1.0; // release to add
    // printf("Lock %i released\n", i);
  }
  __syncthreads();
  cg::thread_block g = cg::this_thread_block();
  // __shared__ cg::experimental::block_tile_memory<4, 1024> shared;
  __shared__ cg::experimental::block_tile_memory<1, 512> shared;
  cg::thread_block thb = cg::experimental::this_thread_block(shared);

  auto tilehalf = cg::experimental::tiled_partition<512>(thb);
  int lane = tilehalf.thread_rank();

  int lanerel = lane + blockIdx.x * blockDim.x / 2 + start;

  int tid = threadIdx.x + blockIdx.x * blockDim.x + start;

  int latecutoff = tid - tilehalf.meta_group_rank() * tilehalf.size();

  if (lanerel >= N || lanerel >= end)
    return;

  int ntid = tid + 1;
  int ltid = lanerel + 1;
  // int nltid = ltid + 1;

  // 513, we want 512 to be ok and then ignore
  //  if (ntid >= N && tilehalf.meta_group_rank() == 0) return;
  if (ltid >= N)
    return;

  const ccdgpu::Scalar3 &a = sm[lanerel]; // s_sortedmin[ltid];
  ccdgpu::Scalar3 b =
      sm[ltid]; // nltid < nbox*blockDim.x ? s_sortedmin[nltid] : sm[ntid
                // int2 val = make_int2(int(a.z), int(b.z));
                // int whocares = queue.push(lanerel+(ntid-tid-1), val);
                // atomicAdd(count, 1);
                // if (lane == 72) printf("a.y: %.3f, b.x %.3f\n", a.y, b.x);
  while (a.y >= b.x) // curr max > following min
  {
    // consider_pair(int(a.z), int(b.z), count, out, guess);
    if (tilehalf.meta_group_rank() == 0) {

      int2 val = make_int2(int(a.z), int(b.z));
      int whocares = queue.push(lanerel + 512 * (ntid - tid - 1), val);
    } else {
      // unsigned ns = 100;
      // for (int i=0; i<10000; i++)
      // __nanosleep(ns);
      int2 res = queue.pop(lanerel + 512 * (ntid - tid - 1));
      // if (res.x < 0) return;

      const MiniBox &ax = mini[res.x];
      const MiniBox &bx = mini[res.y];

      if (does_collide(ax, bx) && is_valid_pair(ax.vertexIds, bx.vertexIds) &&
          !covertex(ax.vertexIds, bx.vertexIds)) {
        add_overlap(res.x, res.y, count, overlaps, guess);
      }
    }
    // atomicAdd(count, 1);
    // return;
    ntid++;
    ltid++;
    // nltid++;
    // if (ntid >= N && tilehalf.meta_group_rank() == 0) return;
    if (ltid >= N)
      return;
    b = sm[ltid]; // nltid < nbox*blockDim.x ? s_sortedmin[nltid] :
    sm[ntid];
  }
}

// BigWorkerQueue
// we will have int2 queue, int start, int end
// we keep going start+tid, atomicAdd the end
// we need to have the sweep function + initially setup first array

__global__ void init_bigworkerqueue(int2 *queue, int N) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N)
    return;

  queue[tid] = make_int2(tid, tid + 1);
}

__global__ void shift_queue_pointers(int *d_N, unsigned *d_start,
                                     unsigned *d_end) {
  *d_start += *d_N;
  *d_start = *d_start % 2000000;
  *d_N = (*d_end - *d_start);
  *d_N = *d_N < 0 ? *d_end + 2000000 - *d_start : *d_N;
}

__global__ void sweepqueue(int2 *queue, const Aabb *boxes, int *count,
                           int guess, int *d_N, int N, int N0, unsigned *start,
                           unsigned *end, int2 *overlaps) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N)
    return;

  // queue[tid] = make_int2(tid, tid + 1);

  int qid = (tid + *start) % 2000000;

  const int2 check = queue[qid];

  if (check.y >= N0)
    return;

  const Aabb &a = boxes[check.x];
  Aabb b = boxes[check.y];

  int inc = 0;
  int MAX_CHECK = N0;
  while (a.max.x >= b.min.x) {

    if (does_collide(a, b) && !covertex(a.vertexIds, b.vertexIds)) {
      add_overlap(a.id, b.id, count, overlaps, guess);
    }

    inc++;

    // append_queue(check, MAX_CHECK, queue, end);
    if (check.y + inc >= N0)
      return;

    if (inc == MAX_CHECK) {
      append_queue(check, MAX_CHECK, queue, end);
      return;
    }

    b = boxes[check.y + inc];
  }
}

//////////////
// __global__ void create_rankbox(Aabb * boxes, RankBox * rankboxes, int N)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;

//     if (tid >= N) return;

//     rankboxes[tid].aabb = &boxes[tid];
// }

// __global__ void register_rank_x(RankBox * rankboxes, int N)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= N) return;
//     rankboxes[tid].rank_x = tid;
// }

// __global__ void register_rank_y(RankBox * rankboxes, int N)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= N) return;
//     rankboxes[tid].rank_y = tid;
// }

// __device__ ull cantor(ull x, ull y)
// {
//     return (x+y)/2 * (x+y+1) + y;
// }

// __global__ void assign_rank_c(RankBox * rankboxes, int N)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= N) return;
//     rankboxes[tid].rank_c = cantor(rankboxes[tid].rank_x,
//     rankboxes[tid].rank_y);
// }

// __global__ void build_checker2(const RankBox * const rankboxes, int2 *
// overlaps, int N, int * count, int guess)
// {
//     extern __shared__ RankBox s_rank[];

//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int ltid = threadIdx.x;

//     if (tid >= N) return;

//     int ntid = tid + 1;
//     int nltid = ltid + 1;

//     if (ntid >= N) return;

//     s_rank[ltid] = rankboxes[tid];

//     __syncthreads();

//     const RankBox a = s_rank[ltid];
//     RankBox b = rankboxes[ntid]; //nltid < blockDim.x ? s_rank[nltid] :
//     rankboxes[ntid];

//     while (a.aabb->max.x  >= b.aabb->min.x || a.aabb->max.y  >=
//     b.aabb->min.y) // curr max > following min
//     {
//         if(b.rank_x <= a.rank_x && b.rank_y <= a.rank_y &&
//             // does_collide(a.aabb, b.aabb) &&
//             //  a.aabb->max.z >= b.aabb->min.z && a.aabb->min.z <=
//             b.aabb->max.z && !covertex(a.aabb->vertexIds, b.aabb->vertexIds)
//         )
//         {

//             add_overlap(a.aabb->id, b.aabb->id, count, overlaps, guess);
//         }

//         ntid++;
//         nltid++;
//         if (ntid >= N) return;
//         b =  rankboxes[ntid]; //nltid < blockDim.x ? s_rank[nltid] :
//         rankboxes[ntid];
//     }
// }

// __global__ void print_stats(RankBox * rankboxes, int N)
// {
//     for (int i=N-50; i < N; i++)
//     {
//         RankBox & curr = rankboxes[i];
//         printf("id: %i -> rank_x %llu rank_y %llu rank_c %llu\n",
//         curr.aabb->id, curr.rank_x, curr.rank_y, curr.rank_c);
//     }
//     // rankboxes[tid].rank_c = cantor(rankboxes[tid].rank_x,
//     rankboxes[tid].rank_y);
// }