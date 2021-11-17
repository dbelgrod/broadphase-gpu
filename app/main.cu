#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>
#include <unistd.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/edges.h>

#include <gpubf/simulation.cuh>
#include <gpubf/groundtruth.h>
#include <gpubf/util.cuh>
#include <gpubf/klee.cuh>
#include <gpubf/io.cuh>

using namespace std;

int main( int argc, char **argv )
{
    vector<char*> compare;

    const char* filet0 = argv[1];
    const char* filet1 = argv[2];
    
    vector<Aabb> boxes;
    Eigen::MatrixXd vertices_t0;
    Eigen::MatrixXd vertices_t1;
    Eigen::MatrixXi faces; 
    Eigen::MatrixXi edges;

    parseMesh(filet0, filet1, vertices_t0, vertices_t1, faces, edges);
    constructBoxes(vertices_t0, vertices_t1, faces, edges, boxes);
    int N = boxes.size();
    int nbox = 0;
    int parallel = 0;
    bool evenworkload = false;
    int devcount = 1;
    bool pairing = false;

    int o;
    while ((o = getopt (argc, argv, "c:n:b:p:d:WP")) != -1)
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
            case 'b':
                nbox = atoi(optarg);
                break;
            case 'p':
                parallel = stoi(optarg);
                break;
            case 'd':
                devcount = atoi(optarg);
                break;
            case 'W':
                evenworkload = true;
                break;
            case 'P':
                pairing = true;
                break;
        }
    }
    printf("Boxes (N): %i\n", N);
    vector<pair<int,int>> overlaps;
    if (evenworkload)
        run_sweep_pieces(boxes.data(), N, nbox, overlaps, parallel, devcount);
    else if (pairing)
        run_sweep_pairing(boxes.data(), N, nbox, overlaps, parallel, devcount);
    else
        run_sweep_multigpu(boxes.data(), N, nbox, overlaps, parallel, devcount);
    for (auto i : compare)
    {
        // printf("%s\n", i );
        compare_mathematica(overlaps, i);
    }
    cout << endl;

}
