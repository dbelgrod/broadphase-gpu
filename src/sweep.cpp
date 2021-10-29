#include "aabb.h"
#include "timer.hpp"

#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>

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

void add_overlap(const int& xid, const int& yid, atomic<long long>& count, std::vector<int> local_overlaps, long long G)
{
    // int i = atomicAdd(count, 1); //how to do this

    // do x+=y and return the old value of x
    // long long i = count.fetch_add(1);
    
    // int i = count;
    // printf("%i\n", i);
    // if (i < G)
    // {
        // local_overlaps.emplace_back(xid,yid);
    // } 
    // count++;
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

struct sort_boxes //sort_aabb_x
{
    bool operator()(const Aabb &a, const Aabb &b) const {
        return (a.min[0] < b.min[0]);}
};

void sort_along_xaxis(Aabb * boxes, int * box_indices, int N)
{
    // sort box indices by boxes minx val
    sort(execution::par_unseq, box_indices, box_indices + N, sort_indices(boxes));
    // sort boxes by minx val
    sort(execution::par_unseq, boxes, boxes + N, sort_boxes());
}

void merge_local_overlaps(
    const tbb::enumerable_thread_specific<std::vector<int>>& storages,
    std::vector<int>& candidates)
{
    // size up the candidates
    size_t num_candidates = candidates.size();
    for (const auto& local_candidates : storages) {
        num_candidates += local_candidates.size();
    }
    // serial merge!
    candidates.reserve(num_candidates);
    for (const auto& local_candidates : storages) {
        candidates.insert(
            candidates.end(), local_candidates.begin(), local_candidates.end());
    }
}


void sweep(const Aabb * boxes, const int * box_indices, atomic<long long>& count, std::vector<int>& overlaps, int N, long long guess)
{
    tbb::enumerable_thread_specific<std::vector<int>> storages;

    tbb::parallel_for(tbb::blocked_range<int>(0,N), [&](tbb::blocked_range<int> r){
        auto& local_overlaps = storages.local();
        
        for (int i=r.begin(); i<r.end(); i++){
            const Aabb a = boxes[i];
            int inc = i+1;
            if (inc >= N) return;
            Aabb b = boxes[inc];

            while (a.max[0]  >= b.min[0])
            {
                if (
                    does_collide(a,b) &&
                    !covertex(a.vertexIds, b.vertexIds)
                    )
                    {
                        local_overlaps.emplace_back(box_indices[i]);
                        local_overlaps.emplace_back(box_indices[inc]);
                        // add_overlap(box_indices[0], box_indices[inc], count, local_overlaps, guess);
                    }
                inc++;
                if (inc >= N) return;
                b = boxes[inc];
            }
        }
    });

     merge_local_overlaps(storages, overlaps);
}


void run_sweep_cpu(
    Aabb* boxes, 
    int N, int numBoxes, 
    vector<unsigned long>& finOverlaps)
{
    Aabb * og_boxes = new Aabb[N];
    for(int i=0; i<N; ++i)
        og_boxes[i] = boxes[i];
    // sort boxes by xaxis in parallel
    // we will need an index vector
    int * box_indices = new int[N];
    for (int i=0;i<N;i++) {box_indices[i]=i;}
    // printf("Running sort\n");
    sort_along_xaxis(boxes, box_indices, N);
    // printf("Finished sort\n");

    long long guess = 0;
    std::vector<int> overlaps;
    
    atomic<long long> count = 0;

    // printf("Running again...\n");
    guess = count;
    // delete[] overlaps;  //probably dont need
    // delete overlaps;
    // overlaps = new int[2*guess];
    // count = 0;
    // cout << "count: " << count << ", guess: " << guess << endl;
    ccd::Timer timer;
    timer.start();
    sweep(boxes, box_indices, count, overlaps, N, guess);
    timer.stop();
    double total_time = 0;
    total_time += timer.getElapsedTimeInMicroSec();
    printf("Elapsed time: %.6f ms\n", total_time / 1000);

    cout << "Final count: " << overlaps.size() << endl;

    for (size_t i=0; i < overlaps.size() / 2; i++)
    {
        // finOverlaps.push_back(overlaps[i].x);
        // finOverlaps.push_back(overlaps[i].y);
        
        // need to fetch where box is from index first
        const Aabb& a = og_boxes[overlaps[2*i]];
        const Aabb& b = og_boxes[overlaps[2*i+1]];
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
    // delete[] overlaps; 
    delete[] box_indices; 
    delete[] og_boxes; 
}

    // #pragma omp declare reduction (merge : std::vector<long> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    // #pragma omp parallel for num_threads(num_threads),reduction(+:m_narrowPhase), reduction(merge:narrowPhaseValues)

    