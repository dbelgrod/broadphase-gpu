#pragma once

#include <gpubf/aabb.cuh>

// using namespace ccdgpu;

void constructBoxes(const Eigen::MatrixXd &vertices_t0,
                    const Eigen::MatrixXd &vertices_t1,
                    const Eigen::MatrixXi &faces, const Eigen::MatrixXi &edges,
                    vector<ccdgpu::Aabb> &boxes);

void parseMesh(const char *filet0, const char *filet1, Eigen::MatrixXd &V0,
               Eigen::MatrixXd &V1, Eigen::MatrixXi &F, Eigen::MatrixXi &E);