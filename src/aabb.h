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
        unsigned long ref_id;
        float min[3]{};
        float max[3]{};
        Simplex type;

        Aabb(int assignid)
        {
            float tempmax[3] = {1,1,1};
            float tempmin[3] = {0,0,0};
            memcpy(max,tempmax, sizeof(float)*3);
            memcpy(min,tempmin, sizeof(float)*3);
            id = assignid;
            ref_id = assignid;
        };

        Aabb(int assignid, unsigned long reference_id, Simplex assigntype, float* tempmin, float* tempmax)
        {
            memcpy(min, tempmin, sizeof(float)*3);
            memcpy(max ,tempmax, sizeof(float)*3);
            id = assignid;
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