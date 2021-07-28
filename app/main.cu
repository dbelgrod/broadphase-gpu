#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <igl/readOBJ.h>
#include <igl/edges.h>

#include <gpubf/simulation.h>

using namespace std;

void constructBoxes
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    Eigen::MatrixXi& faces, 
    Eigen::MatrixXi& edges, 
    vector<Aabb>& boxes
)
{
   addEdges(vertices_t0, vertices_t1, edges, boxes);
   addVertices(vertices_t0, vertices_t1, boxes);
   addFaces(vertices_t0, vertices_t1, faces, boxes);
}

void parseMesh(const char* filet0, const char* filet1, vector<Aabb>& boxes)
{

    // read in vertices, faces t=0
    Eigen::MatrixXd V0;
    Eigen::MatrixXi F;
    igl::readOBJ(filet0, V0, F);

    // get edges and close file
    Eigen::MatrixXi E;
    igl::edges(F,E);

    // read in vertices, t=1
    // faces should be same F^{t=0} = F^{t=1}
    Eigen::MatrixXd V1;    
    igl::readOBJ(filet1, V1, F);

    constructBoxes(V0, V1, F, E, boxes);
}


int main( int argc, const char* argv[] )
{
    const char* filet0 = argv[argc-2];
    const char* filet1 = argv[argc-1];
    
    vector<Aabb> boxes;
    parseMesh(filet0, filet1, boxes);

    run_simulation(boxes.data(), boxes.size());
}