#pragma once

#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#include <gpubf/aabb.h>
#include <gpubf/collision.h>
// #include <gpubf/util.cuh>

void setup(int devId, int& smemSize, int& threads);

__device__ struct Cell {
    float3 min;
    float3 max;
    Aabb * past; 

    __device__ Cell(Aabb * boxes__x, Aabb * boxes__y, int N)
    {
        min = {boxes__x[0].min.x , boxes__y[0].min.y, -1};
        max = {boxes__x[N-1].max.x , boxes__y[N-1].max.y, -1};
        past = NULL;
    };

    __device__ Cell() = default;

    // if any box is fully covering 1 side of the domain, shrink the cell + set the box as past
    __device__ void Simplify();

    // 
    __device__ void Cut();

};


void run_klee(const Aabb* boxes, int N, int numBoxes, vector<unsigned long>& finOverlaps);

// lets make an isolated 2d case 
// use real data however

// get max/min x,y to create 2d cell
// split on what Chan mentioned (we need to split until we have a slab)
// --> add to queue
// pull from queue (1 thread or whoever does)
// simplify
// collect pairs 
// if more than 1 box in cell --> split again
// else terminate

// we can almost compare this to sweep because 
// instead of just having x by default and y on border (and not check z) we can also check z by hand
// this will give an idea of the speedup from a 2d concept


// we should investigate first why the SAP takes nk = n*n^(2/3) 


//pseudocode
// sort the boxes first the x,y axis and feed in those 2 sets
// __device__ Cell(Aabb * boxes) -> get the big cell size
// __device__ Cut()




