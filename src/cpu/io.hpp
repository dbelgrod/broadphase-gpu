#pragma once

#include <vector>
#include <stq/cpu/aabb.hpp>

namespace stq::cpu {

void constructBoxes(const Eigen::MatrixXd &vertices_t0,
                    const Eigen::MatrixXd &vertices_t1,
                    const Eigen::MatrixXi &edges, const Eigen::MatrixXi &faces,
                    std::vector<Aabb> &boxes, int threads = -1);

void parseMesh(const char *filet0, const char *filet1,
               std::vector<Aabb> &boxes);

} // namespace stq::cpu