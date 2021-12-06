#pragma once

#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/edges.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>
#include <unistd.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <vector>
#include <set>
#include <fstream>
#include <iostream>
// #include <nlohmann/json.hpp>

#include <gpubf/aabb.h>

using namespace std;

namespace ccdcpu {

void constructBoxes
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    Eigen::MatrixXi& faces, 
    Eigen::MatrixXi& edges, 
    vector<Aabb>& boxes
);

void parseMesh(const char* filet0, const char* filet1, vector<Aabb>& boxes);

} //namespace