#pragma once

#include <vector>

#include <Eigen/Core>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/info.h>

namespace stq::cpu {

#ifdef CCD_USE_DOUBLE
typedef double Scalar;
// #warning Using Double
#else
typedef float Scalar;
// #warning Using Float
#endif

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
  const tbb::enumerable_thread_specific<std::vector<Aabb>> &storages,
  std::vector<Aabb> &boxes);

void addEdges(const Eigen::MatrixXd &vertices_t0,
              const Eigen::MatrixXd &vertices_t1, const Eigen::MatrixXi &edges,
              std::vector<Aabb> &boxes);

void addVertices(const Eigen::MatrixXd &vertices_t0,
                 const Eigen::MatrixXd &vertices_t1, std::vector<Aabb> &boxes);

void addFaces(const Eigen::MatrixXd &vertices_t0,
              const Eigen::MatrixXd &vertices_t1, const Eigen::MatrixXi &faces,
              std::vector<Aabb> &boxes);

} // namespace stq::cpu