#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/edges.h>

#include <gpubf/simulation.h>
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
    parseMesh(filet0, filet1, boxes);
    int N = boxes.size();
    int nbox = 0;
    
    int o;
    while ((o = getopt (argc, argv, "c:n:b:")) != -1)
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
        }
    }

    vector<unsigned long> overlaps;
    // int i = 1;
    // while (i < N)
    // {
    //     run_scaling(boxes.data(), i, overlaps);
    //     printf("\n");
    //     i = i << 1;
    // }
    run_sweep(boxes.data(), N, nbox, overlaps);
    for (auto i : compare)
    {
        // printf("%s\n", i );
        compare_mathematica(overlaps, i);
    }
    cout << endl;

}