#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <igl/readOBJ.h>
#include <igl/edges.h>

#include <gpubf/simulation.h>
#include <gpubf/groundtruth.h>

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
    addVertices(vertices_t0, vertices_t1, boxes);
    addEdges(vertices_t0, vertices_t1, edges, boxes);
    addFaces(vertices_t0, vertices_t1, faces, boxes);
}

void parseMesh(const char* filet0, const char* filet1, vector<Aabb>& boxes)
{
    Eigen::MatrixXd V0;
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F;
    igl::readOBJ(filet0, V0, F);
    igl::readOBJ(filet1, V1, F);
    Eigen::MatrixXi E;
    igl::edges(F,E);
    // faces should be same F^{t=0} = F^{t=1}
    constructBoxes(V0, V1, F, E, boxes);
}


int main( int argc, char **argv )
{
    vector<char*> compare;

    const char* filet0 = argv[1];
    const char* filet1 = argv[2];
    
    vector<Aabb> boxes;
    parseMesh(filet0, filet1, boxes);
    int N = boxes.size();
    
    int o;
    while ((o = getopt (argc, argv, "c:n:")) != -1)
    {
        switch (o)
        {
            case 'c':
                optind--;
                for( ;optind < argc && *argv[optind] != '-'; optind++)
                {
                    compare.push_back(argv[optind]);
                    // compare_mathematica(overlaps, argv[optind]); 
                }
                break;
            case 'n':
                N = atoi(optarg);
                break;
        }
    }

    vector<unsigned long> overlaps;
    run_scaling(boxes.data(), N, overlaps);
    for (auto i : compare)
    {
        printf("%s\n", i );
        compare_mathematica(overlaps, i);
    }
}