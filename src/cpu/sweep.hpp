#pragma once

#include <stq/cpu/aabb.hpp>
#include <vector>

namespace stq::cpu {

bool is_face(const std::array<int, 3> &vids);

bool is_edge(const std::array<int, 3> &vids);

bool is_vertex(const std::array<int, 3> &vids);

bool is_valid_pair(const std::array<int, 3> &a, const std::array<int, 3> &b);

void run_sweep_cpu(std::vector<Aabb> &boxes, int &n,
                   std::vector<std::pair<int, int>> &finOverlaps);

void sweep_cpu_single_batch(std::vector<Aabb> &boxes_batching, int &n, int N,
                            std::vector<std::pair<int, int>> &overlaps);

// inline void run_sweep_cpu(std::vector<Aabb> &boxes,
//                           std::vector<std::pair<int, int>> &overlaps) {
//   int N = boxes.size();
//   int n = boxes.size();
//   return run_sweep_cpu(boxes, n, N, overlaps);
// }

void sort_along_xaxis(std::vector<Aabb> &boxes);

} // namespace stq::cpu