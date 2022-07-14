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

// void sort_along_xaxis(std::vector<Aabb> &boxes, std::vector<int>
// &box_indices,
//                       int N) {
//   // sort box indices by boxes minx val
//   tbb::parallel_sort(box_indices.begin(), box_indices.end(),
//                      sort_indices(boxes.data()));
//   // sort boxes by minx val
//   tbb::parallel_sort(boxes.begin(), boxes.end(), sort_boxes);
// }

void sort_along_xaxis(std::vector<Aabb> &boxes) {
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

void sweep(const std::vector<Aabb> &boxes,
           std::vector<std::pair<int, int>> &overlaps, int n) {
  ThreadSpecificOverlaps storages;

  tbb::parallel_for(
    tbb::blocked_range<long>(0l, n), [&](const tbb::blocked_range<long> &r) {
      auto &local_overlaps = storages.local();

      for (long i = r.begin(); i < r.end(); i++) {
        const Aabb &a = boxes[i];

        for (long j = i + 1; j < boxes.size(); j++) {
          const Aabb &b = boxes[j];

          if (a.max[0] < b.min[0]) {
            break;
          }

          if (does_collide(a, b) && is_valid_pair(a.vertexIds, b.vertexIds) &&
              !covertex(a.vertexIds, b.vertexIds)) {
            local_overlaps.emplace_back(a.id, b.id);
          }
        }
      }
    });

  merge_local_overlaps(storages, overlaps);
}

void run_sweep_cpu(std::vector<Aabb> &boxes, int &n,
                   std::vector<std::pair<int, int>> &finOverlaps) {
  finOverlaps.clear();

  // sort_along_xaxis(boxes, N);

  while (1) {
    try {
      sweep(boxes, finOverlaps, n);
      break;
    } catch (std::bad_alloc &ex) {
      finOverlaps.clear();
      n /= 2;
      spdlog::warn("Out of memory, trying n: {:d}", n);
    }
  }
  spdlog::trace("Final count: {:d}", finOverlaps.size());

  spdlog::trace("Total(filt.) overlaps: {:d}", finOverlaps.size());
}

void sweep_cpu_single_batch(std::vector<Aabb> &boxes_batching, int &n, int N,
                            std::vector<std::pair<int, int>> &overlaps) {
  overlaps.clear();
  if (boxes_batching.size() == 0)
    return;
  if (boxes_batching.size() == N)
    sort_along_xaxis(boxes_batching);

  run_sweep_cpu(boxes_batching, n, overlaps);
  spdlog::debug("N {:d}, boxes {:d}, overlaps {:d}, tot {:d}", n,
                boxes_batching.size(), overlaps.size(), N);
  boxes_batching.erase(boxes_batching.begin(), boxes_batching.begin() + n);
  n = std::min(static_cast<int>(boxes_batching.size()), n);
}

} // namespace stq::cpu