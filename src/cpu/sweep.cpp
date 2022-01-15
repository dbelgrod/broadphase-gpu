#include <gpubf/sweep.hpp>

#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

#include <algorithm> // std::sort
#include <execution>
#include <iostream> // std::cout
#include <vector>   // std::vector

namespace ccdcpu {

// typedef StructAlignment(32) std::array<_simd, 6> SimdObject;

bool is_face(const int *vids) { return vids[2] >= 0; };

bool is_edge(const int *vids) { return vids[2] < 0 && vids[1] >= 0; };

bool is_vertex(const int *vids) { return vids[2] < 0 && vids[1] < 0; };

bool is_valid_pair(const int *a, const int *b) {
  return (is_vertex(a) && is_face(b)) || (is_face(a) && is_vertex(b)) ||
         (is_edge(a) && is_edge(b));
};

bool does_collide(const Aabb &a, const Aabb &b) {
  return
      //    a.max[0] >= b.min[0] && a.min[0] <= b.max[0] && //ignore x axis
      a.max[1] >= b.min[1] && a.min[1] <= b.max[1] && a.max[2] >= b.min[2] &&
      a.min[2] <= b.max[2];
}

bool covertex(const int *a, const int *b) {

  return a[0] == b[0] || a[0] == b[1] || a[0] == b[2] || a[1] == b[0] ||
         a[1] == b[1] || a[1] == b[2] || a[2] == b[0] || a[2] == b[1] ||
         a[2] == b[2];
}

// https://stackoverflow.com/questions/3909272/sorting-two-corresponding-arrays
class sort_indices {
private:
  Aabb *mparr;

public:
  sort_indices(Aabb *parr) : mparr(parr) {}
  bool operator()(int i, int j) const {
    return (mparr[i].min[0] < mparr[j].min[0]);
  }
};

struct // sort_aabb_x
{
  bool operator()(Aabb a, Aabb b) const { return (a.min[0] < b.min[0]); }
} sort_boxes;

void sort_along_xaxis(vector<Aabb> &boxes, vector<int> &box_indices, int N) {
  // sort box indices by boxes minx val
  sort(execution::par_unseq, box_indices.begin(), box_indices.end(),
       sort_indices(boxes.data()));
  // sort boxes by minx val
  sort(execution::par_unseq, boxes.begin(), boxes.end(), sort_boxes);
}

void sort_along_xaxis(vector<Aabb> &boxes, int N) {
  sort(execution::par_unseq, boxes.begin(), boxes.end(), sort_boxes);
}

void merge_local_overlaps(
    const tbb::enumerable_thread_specific<std::vector<std::pair<int, int>>>
        &storages,
    std::vector<std::pair<int, int>> &overlaps) {
  overlaps.clear();
  size_t num_overlaps = overlaps.size();
  for (const auto &local_overlaps : storages) {
    num_overlaps += local_overlaps.size();
  }
  // serial merge!
  overlaps.reserve(num_overlaps);
  for (const auto &local_overlaps : storages) {
    overlaps.insert(overlaps.end(), local_overlaps.begin(),
                    local_overlaps.end());
  }
}

// void sweep(const vector<Aabb> &boxes, const vector<int> &box_indices,
//  vector<std::pair<int, int>> &overlaps, int N) {
void sweep(const vector<Aabb> &boxes, vector<std::pair<int, int>> &overlaps,
           int N) {
  tbb::enumerable_thread_specific<std::vector<std::pair<int, int>>> storages;
  tbb::combinable<int> incrementer;
  tbb::combinable<Aabb> boxer;
  // tbb::enumerable_thread_specific<std::vector<int>> increments;

  // tbb::parallel_for(tbb::blocked_range<int>(0,N), [&](const
  // tbb::blocked_range<int>& r){

  // for (int i=r.begin(); i<r.end(); i++){

  tbb::parallel_for(0, N, 1, [&](int &i) {
    const Aabb a = boxes[i];
    int inc = i + 1;
    if (inc >= N)
      return;
    Aabb b = boxes[inc];

    // local_overlaps.emplace_back(1, 2);
    while (a.max[0] >= b.min[0]) //&& inc-i <=1)
    {
      if (does_collide(a, b) && is_valid_pair(a.vertexIds, b.vertexIds) &&
          !covertex(a.vertexIds, b.vertexIds)) {
        auto &local_overlaps = storages.local();
        // local_overlaps.emplace_back(box_indices[i], box_indices[inc]);
        local_overlaps.emplace_back(a.id, b.id);
      }
      inc++;
      if (inc >= N)
        return;
      b = boxes[inc];
    }
    // }
  });

  merge_local_overlaps(storages, overlaps);
}

void run_sweep_cpu(vector<Aabb> &boxes, int N, int numBoxes,
                   vector<pair<int, int>> &finOverlaps) {
  // vector<Aabb> boxes_cpy;
  // boxes_cpy.reserve(N);
  // for (int i = 0; i < N; ++i)
  //   boxes_cpy.push_back(boxes[i]);
  // sort boxes by xaxis in parallel
  // we will need an index vector
  // vector<int> box_indices;
  // box_indices.reserve(N);
  // for (int i = 0; i < N; i++) {
  //   box_indices.push_back(i);
  // }
  finOverlaps.clear();

  printf("Running sort\n");
  // sort_along_xaxis(boxes_cpy, box_indices, N);
  sort_along_xaxis(boxes, N);
  printf("Finished sort\n");

  // std::vector<std::pair<int, int>> overlaps;

  // sweep(boxes_cpy, box_indices, overlaps, N);
  sweep(boxes, finOverlaps, N);
  printf("Final count: %i\n", finOverlaps.size());

  // for (size_t i = 0; i < overlaps.size(); i++) {

  //   // need to fetch where box is from index first
  //   const Aabb &a = boxes[overlaps[i].first];
  //   const Aabb &b = boxes[overlaps[i].second];
  //   if (is_vertex(a.vertexIds) && is_face(b.vertexIds))
  //     finOverlaps.emplace_back(overlaps[i].first, overlaps[i].second);
  //   else if (is_face(a.vertexIds) && is_vertex(b.vertexIds))
  //     finOverlaps.emplace_back(overlaps[i].second, overlaps[i].first);
  //   else if (is_edge(a.vertexIds) && is_edge(b.vertexIds)) {
  //     int minnow = min(overlaps[i].first, overlaps[i].second);
  //     int maxxer = max(overlaps[i].first, overlaps[i].second);
  //     finOverlaps.emplace_back(minnow, maxxer);
  //   }
  // }
  printf("Total(filt.) overlaps: %lu\n", finOverlaps.size());
  // delete[] box_indices;
  // delete[] og_boxes;
}

// #pragma omp declare reduction (merge : std::vector<long> :
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

// #pragma omp parallel for num_threads(num_threads),reduction(+:m_narrowPhase),
// reduction(merge:narrowPhaseValues)

} // namespace ccdcpu