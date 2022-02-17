#pragma once

#include <vector>

#include <Eigen/Core>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/info.h>

#ifdef CCD_USE_DOUBLE
typedef double Scalar;
// #warning Using Double
#else
typedef float Scalar;
// #warning Using Float
#endif

namespace ccd::cpu {

static const int CPU_THREADS = std::min(tbb::info::default_concurrency(), 64);

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
  const tbb::enumerable_thread_specific<tbb::concurrent_vector<Aabb>> &storages,
  std::vector<Aabb> &boxes);

void addEdges(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
              Eigen::MatrixXi &edges, std::vector<Aabb> &boxes);

void addVertices(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
                 std::vector<Aabb> &boxes);

void addFaces(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
              Eigen::MatrixXi &faces, std::vector<Aabb> &boxes);

} // namespace ccd::cpu