#pragma once

#include <gpubf/aabb.hpp>
#include <vector>

using namespace std;

namespace ccdcpu {

bool is_face(const int* vids);

bool is_edge(const int* vids);

bool is_vertex(const int* vids);

bool is_valid_pair(const int* a, const int* b);

void run_sweep_cpu(
    vector<Aabb>& boxes, 
    int N, int numBoxes, 
    vector<pair<long,long>>& finOverlaps);

} //namespace