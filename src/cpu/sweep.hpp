#pragma once

#include <stq/cpu/aabb.hpp>
#include <vector>

namespace stq::cpu {

bool is_face(const std::array<int, 3> &vids);

bool is_edge(const std::array<int, 3> &vids);

bool is_vertex(const std::array<int, 3> &vids);

bool is_valid_pair(const std::array<int, 3> &a, const std::array<int, 3> &b);

void run_sweep_cpu(std::vector<Aabb> &boxes, int N, int numBoxes,
                   std::vector<std::pair<int, int>> &finOverlaps);

inline void run_sweep_cpu(std::vector<Aabb> &boxes,
                          std::vector<std::pair<int, int>> &overlaps) {
  return run_sweep_cpu(boxes, boxes.size(), boxes.size(), overlaps);
}

} // namespace stq::cpu