#pragma once

#include <bits/stdc++.h>

// need this to get tiled_partition > 32 threads
#define _CG_ABI_EXPERIMENTAL // enable experimental API

#include <cooperative_groups.h>

// #include <stq/gpu/aabb.cuh>
#include <stq/gpu/collision.cuh>

namespace cg = cooperative_groups;

namespace stq::gpu {

// #include <cuda/atomic>

// __global__ struct SweepMarker {
//   int id;
//   Scalar x, length;

//   __device__ SweepMarker(int aabbId, Aabb &a) {
//     id = aabbId;
//     x = a.min.x;
//     length = a.max.x - x;
//   }

//   __device__ SweepMarker() = default;
// };

__global__ void build_index(Aabb *boxes, int N, int *index);
__global__ void print_sort_axis(Aabb *axis, int C);
__global__ void retrieve_collision_pairs(const Aabb *const boxes, int *count,
                                         int2 *overlaps, int N, int guess,
                                         int nbox, int start = 0,
                                         int end = INT_MAX);
__global__ void print_overlap_start(int2 *overlaps);

// for balancing
__global__ void build_checker(Scalar3 *sortedmin, int2 *outpairs, int N,
                              int *count, int guess);
__global__ void create_ds(Aabb *boxes, Scalar2 *sortedmin, MiniBox *mini, int N,
                          Dimension axis);
__global__ void retrieve_collision_pairs2(const MiniBox *const mini, int *count,
                                          int2 *inpairs, int2 *overlaps, int N,
                                          int guess);
__global__ void calc_variance(Aabb *boxes, Scalar3 *var, int N, Scalar3 *mean);
__global__ void calc_mean(Aabb *boxes, Scalar3 *mean, int N);
__global__ void twostage_queue(Scalar2 *sm, const MiniBox *const mini,
                               int2 *overlaps, int N, int *count, int guess,
                               int start = 0, int end = INT_MAX);

// for pairing
__global__ void create_ds(Aabb *boxes, Scalar3 *sortedmin, MiniBox *mini, int N,
                          Scalar3 *mean);
__global__ void assign_rank_c(RankBox *rankboxes, int N);
__global__ void register_rank_y(RankBox *rankboxes, int N);
__global__ void register_rank_x(RankBox *rankboxes, int N);
__global__ void create_rankbox(Aabb *boxes, RankBox *rankboxes, int N);
__global__ void build_checker2(const RankBox *const rankboxes, int2 *out, int N,
                               int *count, int guess);
__global__ void print_stats(RankBox *rankboxes, int N);

__global__ void init_bigworkerqueue(int2 *queue, int N);
// __global__ void sweepqueue(int2 * queue, const Aabb * boxes, int *
// count, int guess, int N, int TotBoxes, int start, unsigned * end, int2 *
// overlaps);
__global__ void sweepqueue(int2 *queue, const Aabb *boxes, int *count,
                           int guess, int *d_N, int N, int N0, unsigned *start,
                           unsigned *end, int2 *overlaps);
__global__ void shift_queue_pointers(int *d_N, unsigned *d_start,
                                     unsigned *d_end);

} // namespace stq::gpu