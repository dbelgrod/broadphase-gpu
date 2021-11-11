#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <Eigen/Core>

using namespace std;

typedef enum { VERTEX, FACE, EDGE }  Simplex;

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

