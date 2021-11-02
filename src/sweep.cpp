#include "aabb.h"
#include "timer.hpp"

#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/combinable.h>

#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <execution> 


// typedef StructAlignment(32) std::array<_simd, 6> SimdObject;

bool does_collide(const Aabb& a, const Aabb& b)
{
    return 
    //    a.max[0] >= b.min[0] && a.min[0] <= b.max[0] && //ignore x axis
            a.max[1] >= b.min[1] && a.min[1] <= b.max[1] &&
            a.max[2] >= b.min[2] && a.min[2] <= b.max[2];
}

bool covertex(const int* a, const int* b) {
    
    return a[0] == b[0] || a[0] == b[1] || a[0] == b[2] || 
        a[1] == b[0] || a[1] == b[1] || a[1] == b[2] || 
        a[2] == b[0] || a[2] == b[1] || a[2] == b[2];
}

// https://stackoverflow.com/questions/3909272/sorting-two-corresponding-arrays
class sort_indices
{
   private:
     Aabb* mparr;
   public:
     sort_indices(Aabb* parr) : mparr(parr) {}
     bool operator()(int i, int j) const { return (mparr[i].min[0] < mparr[j].min[0]);}
};

struct //sort_aabb_x
{
    bool operator()(Aabb a,Aabb b) const {
        return (a.min[0] < b.min[0]);}
} sort_boxes;

void sort_along_xaxis(vector<Aabb>& boxes, vector<int>& box_indices, int N)
{
    // sort box indices by boxes minx val
    sort(execution::par_unseq, box_indices.begin(), box_indices.end(), sort_indices(boxes.data()));
    // sort boxes by minx val
    sort(execution::par_unseq, boxes.begin(), boxes.end(), sort_boxes);
}

void merge_local_overlaps(
    const tbb::enumerable_thread_specific<std::vector<std::pair<int,int>>>& storages,
    std::vector<std::pair<int,int>>& overlaps)
{
    overlaps.clear();
    // ask about changing number of boxes per thread !!!
    // tbb::parallel_for(0, queries.size(), 1, [&](int i){
    // // body of the for loop using index i
    //     }); 
    // size up the overlaps
    size_t num_overlaps = overlaps.size();
    for (const auto& local_overlaps : storages) {
        num_overlaps += local_overlaps.size();
    }
    // serial merge!
    overlaps.reserve(num_overlaps);
    for (const auto& local_overlaps : storages) {
        overlaps.insert(
            overlaps.end(), local_overlaps.begin(), local_overlaps.end());
    }
}

void sweep(const vector<Aabb> &boxes, const vector<int>& box_indices, vector<std::pair<int,int>>& overlaps, int N)
{
    tbb::enumerable_thread_specific<std::vector<std::pair<int,int>>> storages;
    tbb::combinable<int> incrementer;
    tbb::combinable<Aabb> boxer;
    // tbb::enumerable_thread_specific<std::vector<int>> increments;

    // tbb::parallel_for(tbb::blocked_range<int>(0,N), [&](const tbb::blocked_range<int>& r){
        
    
        // for (int i=r.begin(); i<r.end(); i++){
            
    tbb::parallel_for(0, N, 1, [&](int & i)    {
            const Aabb a = boxes[i];
            int inc = i+1;
            if (inc >= N) return;
            Aabb b = boxes[inc];

            // local_overlaps.emplace_back(1, 2);
            while (a.max[0]  >= b.min[0] ) //&& inc-i <=1)
            {
                if (
                    does_collide(a,b) &&
                    !covertex(a.vertexIds, b.vertexIds)
                    )
                    {
                        auto& local_overlaps = storages.local();
                        local_overlaps.emplace_back(box_indices[i], box_indices[inc]);
                        // sleep(1);
                    }
                inc++;
                if (inc >= N) return;
                b = boxes[inc];
            }
        // }
    });

     merge_local_overlaps(storages, overlaps);
}


void run_sweep_cpu(
    vector<Aabb>& boxes, 
    int N, int numBoxes, 
    vector<unsigned long>& finOverlaps)
{
    vector<Aabb> og_boxes;
    og_boxes.reserve(N);
    copy(boxes.begin(), boxes.end(), og_boxes.begin());
    // for(int i=0; i<N; ++i)
    //     og_boxes[i] = boxes[i];
    // sort boxes by xaxis in parallel
    // we will need an index vector
    vector<int> box_indices;
    box_indices.reserve(N);
    for (int i=0;i<N;i++) {box_indices.push_back(i);}

    printf("Running sort\n");
    sort_along_xaxis(boxes, box_indices, N);
    printf("Finished sort\n");

    std::vector<std::pair<int,int>> overlaps;
      
    ccd::Timer timer;
    timer.start();
    sweep(boxes, box_indices, overlaps, N);
    timer.stop();
    double total_time = 0;
    total_time += timer.getElapsedTimeInMicroSec();
    printf("Elapsed time: %.6f ms\n", total_time / 1000);
    printf("Final count: %i\n", overlaps.size() );
    

    for (size_t i=0; i < overlaps.size(); i++)
    {
        // finOverlaps.push_back(overlaps[i].x);
        // finOverlaps.push_back(overlaps[i].y);
        
        // need to fetch where box is from index first
        const Aabb& a = og_boxes[overlaps[i].first];
        const Aabb& b = og_boxes[overlaps[i].second];
        if (a.type == Simplex::VERTEX && b.type == Simplex::FACE)
        {
            finOverlaps.push_back(a.ref_id);
            finOverlaps.push_back(b.ref_id);
        }
        else if (a.type == Simplex::FACE && b.type == Simplex::VERTEX)
        {
            finOverlaps.push_back(b.ref_id);
            finOverlaps.push_back(a.ref_id);
        }
        else if (a.type == Simplex::EDGE && b.type == Simplex::EDGE)
        {
            finOverlaps.push_back(min(a.ref_id, b.ref_id));
            finOverlaps.push_back(max(a.ref_id, b.ref_id));
        }
    }
    printf("Total(filt.) overlaps: %lu\n", finOverlaps.size() / 2);
    // delete[] box_indices; 
    // delete[] og_boxes; 
}

    // #pragma omp declare reduction (merge : std::vector<long> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    // #pragma omp parallel for num_threads(num_threads),reduction(+:m_narrowPhase), reduction(merge:narrowPhaseValues)

    