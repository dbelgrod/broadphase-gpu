#pragma once

#include <gpubf/sweep.cuh>
// #include <gpubf/util.cuh>

void run_collision_counter(Aabb* boxes, int N);
void run_scaling(const Aabb* boxes, int N, int nBox, vector<unsigned long>& overlaps);
void run_sweep(const Aabb* boxes, int N, int numBoxes, vector<unsigned long>& overlaps);