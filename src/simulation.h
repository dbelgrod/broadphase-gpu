#pragma once

#include <gpubf/collision.h>

void run_collision_counter(Aabb* boxes, int N);
void run_scaling(const Aabb* boxes, int N, vector<unsigned long>& overlaps);