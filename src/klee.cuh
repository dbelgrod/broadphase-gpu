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
    Aabb past;
    Aabb * boxes; 
    int Nboxes;

    __device__ Cell(Aabb * boxes__x, Aabb * boxes__y, int N)
    {
        min = {boxes__x[0].min.x , boxes__y[0].min.y, -1};
        max = {boxes__x[N-1].max.x , boxes__y[N-1].max.y, -1};
        // past = NULL;
        boxes = boxes__x;
        Nboxes = N;
    };

    __device__ Cell(Cell* parent, float left, float right, float upper, float lower)
    {
        min = {left, lower, -1};
        max = {right, upper, -1};
        past = parent->past;
        boxes = parent->boxes;
        Nboxes = 0;
        
        for (size_t i = 0; i < parent->Nboxes; i++)
        {
            if (parent->boxes[i].min.x > left && parent->boxes[i].max.x < right && 
                 parent->boxes[i].min.y > lower && parent->boxes[i].max.y < upper)
            {
                boxes[Nboxes] = parent->boxes[i];
                Nboxes++;
            }
        }
    };

    //__device__ Cell() = default;

    // if any box is fully covering 1 side of the domain, shrink the cell + set the box as past
    __device__ void Simplify();

    // 
    __device__ void Cut()
    {
        //get the middle x, cut it halfway on y and at its points
        past = boxes[Nboxes/2];
        Cell cellLU(this, min.x, past.min.x, max.y, (past.max.y-past.min.y)/2); //Cell, Left, Right, Upper, Lower
        Cell cellLL(this, min.x, past.min.x, (past.max.y-past.min.y)/2, min.y);
        Cell cellRU(this, past.max.x, max.x, max.y, (past.max.y-past.min.y)/2);
        Cell cellRL(this, past.max.x, max.x, (past.max.y-past.min.y)/2, min.y);
    };

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




