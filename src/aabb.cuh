#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <Eigen/Core>


using namespace std;

typedef enum { VERTEX, FACE, EDGE }  Simplex;
typedef enum { x, y, z }  Dimension;
typedef unsigned long long int ull;


__global__ class Aabb {
    public:
        int id;
        float3 min;
        float3 max;
        int3 vertexIds;
        int ref_id;
        // Simplex type;
        // float buffer;
       

        // Aabb(int assignid)
        // {
        //     float4 tempmax[3] = {1,1,1};
        //     float4 tempmin[3] = {0,0,0};
        //     memcpy(max,tempmax, sizeof(half)*3);
        //     memcpy(min,tempmin, sizeof(half)*3);
        // };

        Aabb(int assignid, int reference_id, Simplex assigntype, int * vids, float* tempmin, float* tempmax)
        {
            min = make_float3(tempmin[0], tempmin[1], tempmin[2]);
            max = make_float3(tempmax[0], tempmax[1], tempmax[2]);
            vertexIds = make_int3(vids[0], vids[1], vids[2]);
            // memcpy(min, 	__float2half(tempmin), sizeof(__half)*3);
            // memcpy(max ,	__float2half(tempmax), sizeof(__half)*3);
            id = assignid;
            ref_id = reference_id;
            // type = assigntype;
        };

        Aabb() = default;
};

void addEdges
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    Eigen::MatrixXi& edges, 
    vector<Aabb>& boxes
);

void addVertices
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    vector<Aabb>& boxes
);

void addFaces
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    Eigen::MatrixXi& faces, 
    vector<Aabb>& boxes
);

// bool is_face = [](Aabb& x)
// bool is_edge = [](Aabb& x){return x.vertexIds.z < 0 && x.vertexIds.y >= 0 ;};
// bool is_vertex = [](Aabb& x){return x.vertexIds.z < 0  && x.vertexIds.y < 0;};

__host__ __device__ bool is_face(const Aabb& x);
__host__ __device__ bool is_edge(const Aabb& x);
__host__ __device__ bool is_vertex(const Aabb& x);
__host__ __device__ bool is_valid_pair(const Aabb& x, const Aabb& y);
__host__ __device__ bool is_face(const int3& vids);
__host__ __device__ bool is_edge(const int3& vids);
__host__ __device__ bool is_vertex(const int3& vids);
__host__ __device__ bool is_valid_pair(const int3& a, const int3& b);

__global__ class MiniBox  {
    public:
        float2 min; //only y,z coord
        float2 max;
        int3 vertexIds;

    __device__ MiniBox(float* tempmin, float* tempmax, int3 vids)
        {
            min = make_float2(tempmin[0], tempmin[1]);
            max = make_float2(tempmax[0], tempmax[1]);
            vertexIds = vids;
        };

        MiniBox() = default;
};

__global__ class SortedMin {
    public:
        float3 data;
        int3 vertexIds;

    __device__ SortedMin(float _min, float _max, int assignid, int * vids)
        {
            data = make_float3(_min, _max, float(assignid));
            // min = _min;
            // max = _max;
            vertexIds = make_int3(vids[0], vids[1], vids[2]);
            // id = assignid;
        };

    __device__ SortedMin(float _min, float _max, int assignid, int3 vids)
    {
        data = make_float3(_min, _max, float(assignid));
        // min = _min;
        // max = _max;
        vertexIds = vids;
        // id = assignid;
    };

        SortedMin() = default;
};

__global__ class RankBox {
public:
    Aabb * aabb;
    ull rank_x;
    ull rank_y;
    ull rank_c;
};