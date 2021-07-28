#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <Eigen/Core>

using namespace std;

__global__ class Aabb {
    public:
        int id;
        float min[4]{};
        float max[4]{};

        Aabb(int assignid)
        {
            float tempmax[4] = {1,1,1,0};
            float tempmin[4] = {0,0,0,0};
            memcpy(max,tempmax, sizeof(float)*4);
            memcpy(min,tempmin, sizeof(float)*4);
            id = assignid;
        };

        Aabb(int assignid, float* tempmin, float* tempmax)
        {
            memcpy(min, tempmin, sizeof(float)*4);
            memcpy(min,tempmin, sizeof(float)*4);
            id = assignid;
        };

        Aabb() = default;
        // mat << 1, 2, 6, 9,
        //  3, 1, 7, 2;
  
        // std::cout << "Column's maximum: " << std::endl
        // << mat.colwise().maxCoeff() << std::endl;
        // --> 3, 2, 7, 9
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