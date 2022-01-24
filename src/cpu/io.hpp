#pragma once

#include <igl/edges.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <fstream>
#include <set>
#include <vector>
// #include <nlohmann/json.hpp>
#include <gpubf/aabb.hpp>

using namespace std;

namespace ccdcpu {

void constructBoxes(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
                    Eigen::MatrixXi &edges, Eigen::MatrixXi &faces,
                    vector<Aabb> &boxes);

void parseMesh(const char *filet0, const char *filet1, vector<Aabb> &boxes);

} // namespace ccdcpu