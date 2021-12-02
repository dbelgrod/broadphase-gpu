#pragma once

#include <gpubf/sweep.cuh>
#include <gpubf/timer.cuh>
// #include <gpubf/util.cuh>

#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>
#include "tbb/concurrent_vector.h"
#include <tbb/task_group.h>

void run_collision_counter(Aabb* boxes, int N);
// void run_scaling(const Aabb* boxes, int N, int nBox, vector<unsigned long>& overlaps);
// void run_sweep(const Aabb* boxes, int N, int numBoxes, vector<pair<int,int>>& overlaps, int& threads);
void run_sweep_multigpu(const Aabb* boxes, int N, int nbox, vector<pair<int,int>>& finOverlaps, int& threads, int& devcount);

void run_sweep_pieces(const Aabb* boxes, int N, int nbox, vector<pair<int, int>>& finOverlaps, int2*& d_overlaps, int *& d_count, int& threads, int & devcount);

void run_sweep_pairing(const Aabb* boxes, int N, int nbox, vector<pair<int, int>>& finOverlaps, int& threads, int & devcount);

void run_sweep_multigpu_queue(const Aabb* boxes, int N, int nbox, vector<pair<int, int>>& finOverlaps, int& threads, int & devcount);