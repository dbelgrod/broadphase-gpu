#pragma once

#include "aabb.h"
#include <vector>

using namespace std;

void run_sweep_cpu(
    vector<Aabb>& boxes, 
    int N, int numBoxes, 
    vector<unsigned long>& finOverlaps);