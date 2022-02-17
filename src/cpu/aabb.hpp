#pragma once

#include <array>
#include <vector>

#include <Eigen/Core>

namespace stq::cpu {

#ifdef CCD_USE_DOUBLE
typedef double Scalar;
// #warning Using Double
#else
typedef float Scalar;
// #warning Using Float
#endif

using ArrayMax3 =
  Eigen::Array<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1>;

class Aabb {
public:
  int id;
  ArrayMax3 min;
  ArrayMax3 max;
  std::array<int, 3> vertexIds;

  Aabb() = default;

  Aabb(int assignid, const std::array<int, 3> &vids, const ArrayMax3 &tempmin,
       const ArrayMax3 &tempmax)
      : id(assignid), min(tempmin), max(tempmax), vertexIds(vids){};
};

void constructBoxes(const Eigen::MatrixXd &vertices_t0,
                    const Eigen::MatrixXd &vertices_t1,
                    const Eigen::MatrixXi &edges, const Eigen::MatrixXi &faces,
                    std::vector<Aabb> &boxes, double inflation_radius = 0);

void addVertices(const Eigen::MatrixXd &vertices_t0,
                 const Eigen::MatrixXd &vertices_t1, std::vector<Aabb> &boxes,
                 double inflation_radius = 0);

void addEdges(const std::vector<Aabb> &vertex_boxes,
              const Eigen::MatrixXi &edges, std::vector<Aabb> &boxes);

void addFaces(const std::vector<Aabb> &vertex_boxes,
              const Eigen::MatrixXi &faces, std::vector<Aabb> &boxes);

} // namespace stq::cpu