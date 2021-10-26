#pragma once

#include "aabb.h"

void run_sweep_cpu(
    Aabb* boxes, 
    int N, int numBoxes, 
    vector<unsigned long>& finOverlaps);