#pragma once

#include <Eigen/Core>
#include <assert.h>
#include <cfenv>
#include <cfloat>
#include <functional>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifdef CCD_USE_DOUBLE
typedef double3 Scalar3;
typedef double Scalar;
#warning Using Double
#define make_Scalar3 make_double3
#else
typedef float3 Scalar3;
typedef float Scalar;
#warning Using Float
#define make_Scalar3 make_float3
#endif

using namespace std;
using namespace std::placeholders;

// typedef enum { VERTEX, FACE, EDGE }  Simplex;
typedef enum { x, y, z } Dimension;
typedef unsigned long long int ull;

namespace ccdgpu {

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

  Aabb() = default;
};

void addEdges(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
              Eigen::MatrixXi &edges, vector<Aabb> &boxes);

void addVertices(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
                 vector<Aabb> &boxes);

void addFaces(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
              Eigen::MatrixXi &faces, vector<Aabb> &boxes);

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

} // namespace ccdgpu

__global__ class MiniBox {
public:
#ifdef CCD_USE_DOUBLE
  double2 min; // only y,z coord
  double2 max;
#else
  float2 min; // only y,z coord
  float2 max;
#endif
  int3 vertexIds;

#ifdef CCD_USE_DOUBLE
  __device__ MiniBox(double *tempmin, double *tempmax, int3 vids) {
    min = make_double2(tempmin[0], tempmin[1]);
    max = make_double2(tempmax[0], tempmax[1]);
#else
  __device__ MiniBox(float *tempmin, float *tempmax, int3 vids) {
    min = make_float2(tempmin[0], tempmin[1]);
    max = make_float2(tempmax[0], tempmax[1]);
#endif
    vertexIds = vids;
  };

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
  ccdgpu::Aabb *aabb;
  ull rank_x;
  ull rank_y;
  ull rank_c;
};
