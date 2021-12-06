#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <Eigen/Core>

using namespace std;

namespace ccdcpu {

typedef enum { VERTEX, FACE, EDGE }  Simplex;

class Aabb {
    public:
        // int id;
        double min[3];
        double max[3];
        int vertexIds[3];
        unsigned long ref_id;
        Simplex type;
        // float buffer;
       

        // Aabb(int assignid)
        // {
        //     float4 tempmax[3] = {1,1,1};
        //     float4 tempmin[3] = {0,0,0};
        //     memcpy(max,tempmax, sizeof(half)*3);
        //     memcpy(min,tempmin, sizeof(half)*3);
        // };

        Aabb(int assignid, unsigned long reference_id, Simplex assigntype, int * vids, double* tempmin, double* tempmax)
        {
            for (size_t i = 0; i < 3; i++)
            {
                min[i] = tempmin[i];
                max[i] = tempmax[i];
                vertexIds[i] = vids[i];
            }
            // memcpy(min, 	__float2half(tempmin), sizeof(__half)*3);
            // memcpy(max ,	__float2half(tempmax), sizeof(__half)*3);
            // id = assignid;
            ref_id = reference_id;
            type = assigntype;
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

} //namespace