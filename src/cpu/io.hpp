#pragma once

#include <vector>
#include <gpubf/aabb.hpp>

namespace ccd::cpu {

void constructBoxes(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
                    Eigen::MatrixXi &edges, Eigen::MatrixXi &faces,
                    std::vector<Aabb> &boxes, int threads = -1);

void parseMesh(const char *filet0, const char *filet1,
               std::vector<Aabb> &boxes);

} // namespace ccd::cpu