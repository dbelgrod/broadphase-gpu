#pragma once

#include <gpubf/aabb.cuh>

#include <vector>
#include <utility> // std::std::pair

void run_collision_counter(stq::gpu::Aabb *boxes, int N);
// void run_scaling(const Aabb* boxes, int N, int nBox, std::vector<unsigned
// long>& overlaps); void run_sweep(const Aabb* boxes, int N, int numBoxes,
// std::vector<std::pair<int,int>>& overlaps, int& threads);
void run_sweep_multigpu(const stq::gpu::Aabb *boxes, int N, int nbox,
                        std::vector<std::pair<int, int>> &finOverlaps,
                        int &threads, int &devcount);

void run_sweep_sharedqueue(const stq::gpu::Aabb *boxes, int N, int nbox,
                           std::vector<std::pair<int, int>> &finOverlaps,
                           int2 *&d_overlaps, int *&d_count, int &threads,
                           int &devcount, bool keep_cpu_overlaps = false);

void run_sweep_pairing(const stq::gpu::Aabb *boxes, int N, int nbox,
                       std::vector<std::pair<int, int>> &finOverlaps,
                       int &threads, int &devcount);

void run_sweep_multigpu_queue(const stq::gpu::Aabb *boxes, int N, int nbox,
                              std::vector<std::pair<int, int>> &finOverlaps,
                              int &threads, int &devcount);

void run_sweep_bigworkerqueue(const stq::gpu::Aabb *boxes, int N, int nbox,
                              std::vector<std::pair<int, int>> &finOverlaps,
                              int2 *&d_overlaps, int *&d_count, int &threads,
                              int &devcount);
