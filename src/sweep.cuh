#pragma once

#include <bits/stdc++.h>

#include <gpubf/aabb.cuh>
#include <gpubf/collision.h>



// #include <cuda/atomic>

__global__ struct SweepMarker {
    int id;
    float x,length;

    __device__ SweepMarker(int aabbId, Aabb& a)
    {
        id = aabbId;
        x = a.min.x;
        length = a.max.x - x;
    }

    __device__ SweepMarker() = default;
};

__global__ void build_index(Aabb * boxes, int N, int* index);
__global__ void print_sort_axis(Aabb* axis, int C);
__global__ void retrieve_collision_pairs(const Aabb* const boxes, int * count, int2 * overlaps, int N, int guess, int nbox, int start = 0, int end = INT_MAX);
__global__ void print_overlap_start(int2 * overlaps);

__global__ void build_checker(float3 * sortedmin, int2 * outpairs, int N, int * count, int guess);
__global__ void create_ds(Aabb * boxes, float3 * sortedmin, MiniBox * mini, int N);
// void consider_pair(const int& xid, const int& yid, int * count, int2 * out, int guess);
__global__ void retrieve_collision_pairs2(const MiniBox* const mini, int * count, int2 * inpairs, int2 * overlaps, int N, int guess);

// for pairing
__global__ void assign_rank_c(RankBox * rankboxes, int N);
__global__ void register_rank_y(RankBox * rankboxes, int N);
__global__ void register_rank_x(RankBox * rankboxes, int N);
__global__ void create_rankbox(Aabb * boxes, RankBox * rankboxes, int N);
__global__ void build_checker2(const RankBox * const rankboxes, int2 * out, int N, int * count, int guess);
__global__ void print_stats(RankBox * rankboxes, int N);