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

// https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
unsigned long cantor(unsigned long x, unsigned long y)
{
    return (x + y)*(x + y + 1) / 2 + y;
}


void compare_mathematica(vector<unsigned long> overlaps, const char* jsonPath)
{
    // Get from file
    ifstream in(jsonPath);
    if(in.fail()) 
    {
        printf("%s does not exist", jsonPath);
        return;
    }
    json j_vec = json::parse(in);
    
    set<unsigned long> truePositives = j_vec.get<std::set<unsigned long>>();

    // Transform data to cantor
    set<unsigned long> algoBroadPhase;
    for (size_t i=0; i < overlaps.size(); i+=2)
    {
        algoBroadPhase.emplace(cantor(overlaps[i], overlaps[i+1]));
    }
                                               
    // Get intersection of true positive
    vector<unsigned long> algotruePositives(truePositives.size());
    vector<unsigned long>::iterator it=std::set_intersection (
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

    vector<unsigned long> overlaps;
    // printf("Running sweep\n");
    run_sweep_cpu(boxes.data(), N, nbox, overlaps);
    for (auto i : compare)
    {
        printf("%s\n", i );
        compare_mathematica(overlaps, i);
    }
    exit(0);
}