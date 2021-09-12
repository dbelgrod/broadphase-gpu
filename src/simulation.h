#pragma once

#include <gpubf/sweep.cuh>

void run_collision_counter(Aabb* boxes, int N);
void run_scaling(const Aabb* boxes, int N, int nBox, vector<unsigned long>& overlaps);
void run_sweep(const Aabb* boxes, int N, vector<unsigned long>& overlaps);