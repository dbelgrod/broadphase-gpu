#pragma once

#include <stq/gpu/aabb.cuh>
#include <stq/gpu/memory.cuh>

#include <vector>
#include <utility> // std::std::pair

namespace stq::gpu {

// static const int MAX_OVERLAP_CUTOFF = 1e4;
// static const int MAX_OVERLAP_SIZE = 1e8;

void run_collision_counter(Aabb *boxes, int N);
// void run_scaling(const Aabb* boxes, int N, int nBox, std::vector<unsigned
// long>& overlaps); void run_sweep(const Aabb* boxes, int N, int numBoxes,
// std::vector<std::pair<int,int>>& overlaps, int& threads);
void run_sweep_multigpu(const Aabb *boxes, int N, int nbox,
                        std::vector<std::pair<int, int>> &finOverlaps,
                        int &threads, int &devcount);

void run_sweep_sharedqueue(const Aabb *boxes, MemHandler *memhandle, int N,
                           int nbox,
                           std::vector<std::pair<int, int>> &finOverlaps,
                           int2 *&d_overlaps, int *&d_count, int &threads,
                           int &tidstart, int &devcount,
                           const int memlimit = 0);

void run_sweep_pairing(const Aabb *boxes, int N, int nbox,
                       std::vector<std::pair<int, int>> &finOverlaps,
                       int &threads, int &devcount);

void run_sweep_multigpu_queue(const Aabb *boxes, int N, int nbox,
                              std::vector<std::pair<int, int>> &finOverlaps,
                              int &threads, int &devcount);

void run_sweep_bigworkerqueue(const Aabb *boxes, int N, int nbox,
                              std::vector<std::pair<int, int>> &finOverlaps,
                              int2 *&d_overlaps, int *&d_count, int &threads,
                              int &devcount);

} // namespace stq::gpu