#pragma once

#include <bits/stdc++.h>

#include <gpubf/aabb.cuh>
#include <gpubf/collision.h>

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
__global__ void create_sortedmin(Aabb * boxes, float3 * sortedmin, int N);
// void consider_pair(const int& xid, const int& yid, int * count, int2 * out, int guess);
__global__ void retrieve_collision_pairs2(const Aabb* const boxes, int * count, int2 * inpairs, int2 * overlaps, int N, int guess);