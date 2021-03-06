#include <stq/cpu/sweep.hpp>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <algorithm> // std::sort
#include <vector>    // std::vector

#include <spdlog/spdlog.h>

namespace stq::cpu {

// typedef StructAlignment(32) std::array<_simd, 6> SimdObject;

bool is_face(const std::array<int, 3> &vids) { return vids[2] >= 0; };

bool is_edge(const std::array<int, 3> &vids) {
  return vids[2] < 0 && vids[1] >= 0;
};

bool is_vertex(const std::array<int, 3> &vids) {
  return vids[2] < 0 && vids[1] < 0;
};

bool is_valid_pair(const std::array<int, 3> &a, const std::array<int, 3> &b) {
  return (is_vertex(a) && is_face(b)) || (is_face(a) && is_vertex(b)) ||
         (is_edge(a) && is_edge(b));
}

bool does_collide(const Aabb &a, const Aabb &b) {
  return
    //    a.max[0] >= b.min[0] && a.min[0] <= b.max[0] && //ignore x axis
    a.max[1] >= b.min[1] && a.min[1] <= b.max[1] && a.max[2] >= b.min[2] &&
    a.min[2] <= b.max[2];
}

bool covertex(const std::array<int, 3> &a, const std::array<int, 3> &b) {
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

void sort_along_xaxis(std::vector<Aabb> &boxes, std::vector<int> &box_indices,
                      int N) {
  // sort box indices by boxes minx val
  tbb::parallel_sort(box_indices.begin(), box_indices.end(),
                     sort_indices(boxes.data()));
  // sort boxes by minx val
  tbb::parallel_sort(boxes.begin(), boxes.end(), sort_boxes);
}

void sort_along_xaxis(std::vector<Aabb> &boxes, int N) {
  tbb::parallel_sort(boxes.begin(), boxes.end(), sort_boxes);
}

typedef tbb::enumerable_thread_specific<std::vector<std::pair<int, int>>>
  ThreadSpecificOverlaps;

void merge_local_overlaps(const ThreadSpecificOverlaps &storages,
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

// void sweep(const std::vector<Aabb> &boxes, const std::vector<int>
// &box_indices,
//  std::vector<std::pair<int, int>> &overlaps, int N) {
void sweep(const std::vector<Aabb> &boxes,
           std::vector<std::pair<int, int>> &overlaps, int N) {
  ThreadSpecificOverlaps storages;

  tbb::parallel_for(tbb::blocked_range<long>(0l, N - 1),
                    [&](const tbb::blocked_range<long> &r) {
                      auto &local_overlaps = storages.local();

                      for (long i = r.begin(); i < r.end(); i++) {
                        const Aabb &a = boxes[i];

                        for (long j = i + 1; j < N; j++) {
                          const Aabb &b = boxes[j];

                          if (a.max[0] < b.min[0]) {
                            break;
                          }

                          if (does_collide(a, b) &&
                              is_valid_pair(a.vertexIds, b.vertexIds) &&
                              !covertex(a.vertexIds, b.vertexIds)) {
                            local_overlaps.emplace_back(a.id, b.id);
                          }
                        }
                      }
                    });

  merge_local_overlaps(storages, overlaps);
}

void run_sweep_cpu(std::vector<Aabb> &boxes, int N, int numBoxes,
                   std::vector<std::pair<int, int>> &finOverlaps) {
  // std::vector<Aabb> boxes_cpy;
  // boxes_cpy.reserve(N);
  // for (int i = 0; i < N; ++i)
  //   boxes_cpy.push_back(boxes[i]);
  // sort boxes by xaxis in parallel
  // we will need an index vector
  // std::vector<int> box_indices;
  // box_indices.reserve(N);
  // for (int i = 0; i < N; i++) {
  //   box_indices.push_back(i);
  // }
  finOverlaps.clear();

  // sort_along_xaxis(boxes_cpy, box_indices, N);
  sort_along_xaxis(boxes, N);

  // std::vector<std::pair<int, int>> overlaps;

  // sweep(boxes_cpy, box_indices, overlaps, N);
  sweep(boxes, finOverlaps, N);
  spdlog::trace("Final count: {:d}", finOverlaps.size());

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
  spdlog::trace("Total(filt.) overlaps: {:d}", finOverlaps.size());
  // delete[] box_indices;
  // delete[] og_boxes;
}

// #pragma omp declare reduction (merge : std::vector<long> :
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

// #pragma omp parallel for num_threads(num_threads),reduction(+:m_narrowPhase),
// reduction(merge:narrowPhaseValues)

} // namespace stq::cpu