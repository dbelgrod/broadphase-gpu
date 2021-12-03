#pragma once

#include <gpubf/aabb.cuh>

void constructBoxes
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    Eigen::MatrixXi& faces, 
    Eigen::MatrixXi& edges, 
    vector<Aabb>& boxes
);

void parseMesh(
    const char* filet0, 
    const char* filet1, 
    Eigen::MatrixXd& V0, 
    Eigen::MatrixXd& V1, 
    Eigen::MatrixXi& F, 
    Eigen::MatrixXi& E);