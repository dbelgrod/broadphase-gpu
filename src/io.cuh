#pragma once

#include <vector>

#include <stq/gpu/aabb.cuh>

void constructBoxes(const Eigen::MatrixXd &vertices_t0,
                    const Eigen::MatrixXd &vertices_t1,
                    const Eigen::MatrixXi &edges, const Eigen::MatrixXi &faces,
                    std::vector<stq::gpu::Aabb> &boxes, int threads = -1,
                    stq::gpu::Scalar inflation_radius = 0);

void parseMesh(const char *filet0, const char *filet1, Eigen::MatrixXd &V0,
               Eigen::MatrixXd &V1, Eigen::MatrixXi &F, Eigen::MatrixXi &E);