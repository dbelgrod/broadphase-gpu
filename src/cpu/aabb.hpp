#pragma once

#include <Eigen/Core>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "tbb/concurrent_vector.h"
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#ifdef CCD_USE_DOUBLE
typedef double Scalar;
#warning Using Double
#else
typedef float Scalar;
#warning Using Float
#endif

using namespace std;

namespace ccdcpu {

class Aabb {
public:
  int id;
  Scalar min[3];
  Scalar max[3];
  int vertexIds[3];

  Aabb(int assignid, int *vids, Scalar *tempmin, Scalar *tempmax) {
    id = assignid;
    for (size_t i = 0; i < 3; i++) {
      min[i] = tempmin[i];
      max[i] = tempmax[i];
      vertexIds[i] = vids[i];
    }
  };

  Aabb() = default;
};

void merge_local_boxes(
    const tbb::enumerable_thread_specific<tbb::concurrent_vector<Aabb>>
        &storages,
    std::vector<Aabb> &boxes);

void addEdges(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
              Eigen::MatrixXi &edges, vector<Aabb> &boxes);

void addVertices(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
                 vector<Aabb> &boxes);

void addFaces(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
              Eigen::MatrixXi &faces, vector<Aabb> &boxes);

} // namespace ccdcpu