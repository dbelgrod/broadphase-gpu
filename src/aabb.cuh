#pragma once

#include <algorithm>
#include <vector>

#include <Eigen/Core>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/info.h>

#include <stq/gpu/types.cuh>

namespace stq::gpu {

__global__ class Aabb {
public:
  int id;
  //   double3 block, block2;
  Scalar3 min;
  Scalar3 max;
  int3 vertexIds;
  int ref_id;

  Aabb(int assignid, int reference_id, int *vids, Scalar *tempmin,
       Scalar *tempmax) {
    min = make_Scalar3(tempmin[0], tempmin[1], tempmin[2]);
    max = make_Scalar3(tempmax[0], tempmax[1], tempmax[2]);
    vertexIds = make_int3(vids[0], vids[1], vids[2]);
    id = assignid;
    ref_id = reference_id;
  };

  //   Aabb(int assignid, int reference_id, int *vids, float *tempmin,
  //        float *tempmax) {
  //     min = make_Scalar3(tempmin[0], tempmin[1], tempmin[2]);
  //     max = make_Scalar3(tempmax[0], tempmax[1], tempmax[2]);
  //     vertexIds = make_int3(vids[0], vids[1], vids[2]);
  //     id = assignid;
  //     ref_id = reference_id;
  //   };

  Aabb() = default;
};

void merge_local_boxes(
  const tbb::enumerable_thread_specific<std::vector<Aabb>> &storages,
  std::vector<Aabb> &boxes);

void addEdges(const Eigen::MatrixXd &vertices_t0,
              const Eigen::MatrixXd &vertices_t1, const Eigen::MatrixXi &edges,
              Scalar inflation_radius, std::vector<Aabb> &boxes);

void addVertices(const Eigen::MatrixXd &vertices_t0,
                 const Eigen::MatrixXd &vertices_t1, Scalar inflation_radius,
                 std::vector<Aabb> &boxes);

void addFaces(const Eigen::MatrixXd &vertices_t0,
              const Eigen::MatrixXd &vertices_t1, const Eigen::MatrixXi &faces,
              Scalar inflation_radius, std::vector<Aabb> &boxes);

// bool is_face = [](Aabb& x)
// bool is_edge = [](Aabb& x){return x.vertexIds.z < 0 && x.vertexIds.y >= 0
// ;}; bool is_vertex = [](Aabb& x){return x.vertexIds.z < 0  && x.vertexIds.y
// < 0;};

__host__ __device__ bool is_face(const Aabb &x);
__host__ __device__ bool is_edge(const Aabb &x);
__host__ __device__ bool is_vertex(const Aabb &x);
__host__ __device__ bool is_valid_pair(const Aabb &x, const Aabb &y);
__host__ __device__ bool is_face(const int3 &vids);
__host__ __device__ bool is_edge(const int3 &vids);
__host__ __device__ bool is_vertex(const int3 &vids);
__host__ __device__ bool is_valid_pair(const int3 &a, const int3 &b);

__global__ class MiniBox {
public:
  Scalar2 min; // only y,z coord
  Scalar2 max;
  int3 vertexIds;
  int id;

  __device__ MiniBox(int assignid, Scalar *tempmin, Scalar *tempmax,
                     int3 vids) {
    min = make_Scalar2(tempmin[0], tempmin[1]);
    max = make_Scalar2(tempmax[0], tempmax[1]);
    vertexIds = vids;
    id = assignid;
  };

  //   __device__ MiniBox(float *tempmin, float *tempmax, int3 vids) {
  //     min = make_Scalar2(tempmin[0], tempmin[1]);
  //     max = make_Scalar2(tempmax[0], tempmax[1]);
  //     vertexIds = vids;
  //   };

  MiniBox() = default;
};

__global__ class SortedMin {
public:
  Scalar3 data;
  int3 vertexIds;

  __device__ SortedMin(Scalar _min, Scalar _max, int assignid, int *vids) {
    data = make_Scalar3(_min, _max, Scalar(assignid));
    // min = _min;
    // max = _max;
    vertexIds = make_int3(vids[0], vids[1], vids[2]);
    // id = assignid;
  };

  __device__ SortedMin(Scalar _min, Scalar _max, int assignid, int3 vids) {
    data = make_Scalar3(_min, _max, Scalar(assignid));
    // min = _min;
    // max = _max;
    vertexIds = vids;
    // id = assignid;
  };

  SortedMin() = default;
};

__global__ class RankBox {
public:
  Aabb *aabb;
  ull rank_x;
  ull rank_y;
  ull rank_c;
};

} // namespace stq::gpu