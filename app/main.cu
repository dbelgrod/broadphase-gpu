#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>
#include <unistd.h>
// #include <cuda.h>
// #include <cuda_runtime.h>


#include <gpubf/simulation.cuh>
#include <gpubf/groundtruth.cuh>
#include <gpubf/util.cuh>
// #include <gpubf/klee.cuh>
#include <gpubf/io.cuh>

using namespace std;
using namespace ccdgpu;

bool is_file_exist(const char *fileName)
{
    ifstream infile(fileName);
    return infile.good();
}

int main( int argc, char **argv )
{
    vector<char*> compare;

    char* filet0;
    char* filet1;

    filet0 = argv[1];
    if (is_file_exist(argv[2]))
        filet1 = argv[2];
    else 
        filet1 = argv[1];
    
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
    bool quantumspeed = false;

    int o;
    while ((o = getopt (argc, argv, "c:n:b:p:d:WPQ")) != -1)
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
            case 'Q':
                quantumspeed = true;
                break;
        }
    }
    printf("Boxes (N): %i\n", N);
    vector<pair<int,int>> overlaps;
    int2 * d_overlaps; //device
    int * d_count; //device

    if (evenworkload)
        run_sweep_pieces(boxes.data(), N, nbox, overlaps, d_overlaps, d_count, parallel, devcount);
    else if (pairing)
        run_sweep_pairing(boxes.data(), N, nbox, overlaps, parallel, devcount);
    else if (quantumspeed)
        run_sweep_multigpu_queue(boxes.data(), N, nbox, overlaps, parallel, devcount);
    else
        run_sweep_multigpu(boxes.data(), N, nbox, overlaps, parallel, devcount);

    for (auto i : compare)
    {
        // printf("%s\n", i );
        compare_mathematica(overlaps, i);
    }
    cout << endl;

}
