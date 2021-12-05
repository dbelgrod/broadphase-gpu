#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>
#include <unistd.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <vector>
#include <set>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

// for convenience
using json = nlohmann::json;

#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/edges.h>

#include "../src/aabb.h"
#include "../src/sweep.h"

// #include <gpubf/simulation.h>
// #include <gpubf/groundtruth.h>
// #include <gpubf/util.cuh>
// #include <gpubf/klee.cuh>

#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>

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

    string fn = string(filet0);
    string ext = fn.substr(fn.rfind('.') + 1);

    if(ext == "obj") 
    {
        igl::readOBJ(filet0, V0, F);
        igl::readOBJ(filet1, V1, F);
    }
    else 
    {
        igl::readPLY(filet0, V0, F);
        igl::readPLY(filet1, V1, F);
    }

    
    Eigen::MatrixXi E;
    igl::edges(F,E);
    // faces should be same F^{t=0} = F^{t=1}
    constructBoxes(V0, V1, F, E, boxes);
}




void compare_mathematica(vector<pair<long,long>> overlaps, const char* jsonPath)
{
    // Get from file
    ifstream in(jsonPath);
    if(in.fail()) 
    {
        printf("%s does not exist", jsonPath);
        return;
    }
    json j_vec = json::parse(in);
    
    set<pair<long,long>> truePositives;
    vector<array<long,2>> tmp = j_vec.get<vector<array<long,2>>>();
    for (auto & arr: tmp)
        truePositives.emplace(arr[0], arr[1]);


    // Transform data to cantor
    set<pair<long,long>> algoBroadPhase;
    for (size_t i=0; i < overlaps.size(); i++)
    {
        algoBroadPhase.emplace(overlaps[i].first, overlaps[i].second);
    }
                                               
    // Get intersection of true positive
    vector<pair<long,long>> algotruePositives(truePositives.size());
    vector<pair<long,long>>::iterator it=std::set_intersection (
        truePositives.begin(), truePositives.end(), 
        algoBroadPhase.begin(), algoBroadPhase.end(), algotruePositives.begin());
    algotruePositives.resize(it-algotruePositives.begin());
    
    printf("Contains %lu/%lu TP\n", algotruePositives.size(), truePositives.size());
    return;
}


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
    int parallel = 1;
    while ((o = getopt (argc, argv, "c:n:b:p:")) != -1)
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
        }
    }
    cout<<"default threads "<<tbb::task_scheduler_init::default_num_threads()<<endl;
    tbb::task_scheduler_init init(parallel);
    printf("Running with %i threads\n", parallel);

    vector<pair<long,long>> overlaps;
    // printf("Running sweep\n");
    run_sweep_cpu(boxes, N, nbox, overlaps);
    for (auto i : compare)
    {
        printf("%s\n", i );
        compare_mathematica(overlaps, i);
    }
    exit(0);
}