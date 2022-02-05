#include <gpubf/klee.cuh>

#include <spdlog/spdlog.h>

__global__ void klee_pair_discover(Aabb *boxes__x, Aabb *boxes__y, int *count,
                                   int2 *overlaps, int N, int guess,
                                   int *numBoxes) {
  Cell cell0(boxes__x, boxes__y, N);
  // printf("%i: cell max x %f, cell min x %f, cell max y %f, cell min y %f,
  // cell z %f %f\n", cell0.id, cell0.max.x, cell0.min.x, cell0.max.y,
  // cell0.min.y, cell0.max.z, cell0.min.z);

  Enqueue enqueue;
  enqueue.push(cell0);
  int nitr = 0;
  while (!enqueue.empty()) {
    // cell0.Cut(enqueue);
    Cell c = enqueue.pop();
    // printf("%i: cell max x %f, cell min x %f, cell max y %f, cell min y %f,
    // boxes %i\n", c.id, c.max.x, c.min.x, c.max.y, c.min.y,  c.Nboxes);
    // printf("pop -> curr size: %i\n", enqueue.size());
    // printf("square norm -> %f\n", c.Norm());

    c.Simplify(count, overlaps, guess);
    // printf("simplify -> curr size: %i\n", enqueue.size());

    Cell nextcells[2];
    c.Cut(nextcells);

    // printf("%i:\n", next[0].id);
    // printf("%i: cell max x %f, cell min x %f, cell max y %f, cell min y %f,
    // cell z %f %f\n", nextcells[1].id,  nextcells[1].max.x,
    // nextcells[1].min.x,  nextcells[1].max.y,  nextcells[1].min.y,
    // nextcells[1].max.z,  nextcells[1].min.z); printf("cut -> curr size:
    // %i\n", enqueue.size());
    enqueue.push(nextcells[0]);
    enqueue.push(nextcells[1]);
    // printf("push -> curr size: %i\n", enqueue.size());
    // return;
    nitr++;
    if (c.Nboxes < 3)
      return;
  }
}

//
struct sort_x {
  __host__ __device__ bool operator()(const Aabb &a, const Aabb &b) const {
    return (a.min.x < b.min.x);
  }
};

struct sort_y {
  __host__ __device__ bool operator()(const Aabb &a, const Aabb &b) const {
    return (a.min.y < b.min.y);
  }
};

void run_klee(const Aabb *boxes, int N, int numBoxes,
              vector<unsigned long> &finOverlaps) {
  int devId = 0;
  cudaSetDevice(devId);

  int smemSize;
  int maxBlockSize;

  setup(devId, smemSize, maxBlockSize, numBoxes);

  const int nBoxesPerThread =
    numBoxes ? numBoxes : smemSize / sizeof(Aabb) / maxBlockSize;
  spdlog::trace("Boxes per Thread: {:d}", nBoxesPerThread);
  spdlog::trace("Shared mem alloc: {:d} B",
                nBoxesPerThread * maxBlockSize * sizeof(Aabb));

  int *nbox;
  cudaMalloc((void **)&nbox, sizeof(int));
  cudaMemcpy(nbox, &nBoxesPerThread, sizeof(int), cudaMemcpyHostToDevice);

  finOverlaps.clear();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate boxes to GPU
  Aabb *d_boxes;
  cudaMalloc((void **)&d_boxes, sizeof(Aabb) * N);
  cudaMemcpy(d_boxes, boxes, sizeof(Aabb) * N, cudaMemcpyHostToDevice);

  // Allocate sorted boxes
  Aabb *d_boxes__x;
  Aabb *d_boxes__y;
  cudaMalloc((void **)&d_boxes__x, sizeof(Aabb) * N);
  cudaMalloc((void **)&d_boxes__y, sizeof(Aabb) * N);
  cudaMemcpy(d_boxes__x, boxes, sizeof(Aabb) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_boxes__y, boxes, sizeof(Aabb) * N, cudaMemcpyHostToDevice);

  // Allocate counter to GPU + set to 0 collisions
  int *d_count;
  cudaMalloc((void **)&d_count, sizeof(int));
  reset_counter<<<1, 1>>>(d_count);
  cudaDeviceSynchronize();

  // int SWEEP_BLOCK_SIZE = 1024;
  dim3 block(maxBlockSize);
  int grid_dim_1d = (N / maxBlockSize + 1); /// nBoxesPerThread + 1;
  dim3 grid(grid_dim_1d);
  spdlog::trace("Grid dim (1D): {:d}", grid_dim_1d);
  spdlog::trace("Box size: {:d}", sizeof(Aabb));

  float milliseconds = 0;

  cudaEventRecord(start);
  thrust::sort_by_key(thrust::device, d_boxes, d_boxes + N, d_boxes__x,
                      sort_x());
  cudaMemcpy(d_boxes, boxes, sizeof(Aabb) * N, cudaMemcpyHostToDevice);
  thrust::sort_by_key(thrust::device, d_boxes, d_boxes + N, d_boxes__y,
                      sort_y());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  spdlog::trace("Elapsed time for sort: {:.6f} ms", milliseconds);

  // Find overlapping pairs
  int guess = 400 * N;
  int2 *d_overlaps;
  cudaMalloc((void **)&d_overlaps, sizeof(int2) * (guess));

  cudaEventRecord(start);
  klee_pair_discover<<<1, 1, 0>>>(d_boxes__x, d_boxes__y, d_count, d_overlaps,
                                  N, guess, nbox);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  spdlog::trace("Elapsed time for findoverlaps: {:.6f} ms", milliseconds);

  int count;
  cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  spdlog::trace("Elapsed time: {:.6f} ms", milliseconds);
  spdlog::trace("Collisions: {:d}", count);
  spdlog::trace("Elapsed time: {:.9f} ms/collision", milliseconds / count);
  spdlog::trace("Boxes: {:d}", N);
  spdlog::trace("Elapsed time: {:.9f} ms/box", milliseconds / N);

  int2 *overlaps = (int2 *)malloc(sizeof(int2) * (count));
  cudaMemcpy(overlaps, d_overlaps, sizeof(int2) * (count),
             cudaMemcpyDeviceToHost);

  cudaFree(d_overlaps);
  for (size_t i = 0; i < count; i++) {
    finOverlaps.push_back(overlaps[i].x);
    finOverlaps.push_back(overlaps[i].y);

    // const Aabb& a = boxes[overlaps[i].x];
    // const Aabb& b = boxes[overlaps[i].y];
    // if (a.type == Simplex::VERTEX && b.type == Simplex::FACE)
    // {
    //     finOverlaps.push_back(a.ref_id);
    //     finOverlaps.push_back(b.ref_id);
    // }
    // else if (a.type == Simplex::FACE && b.type == Simplex::VERTEX)
    // {
    //     finOverlaps.push_back(b.ref_id);
    //     finOverlaps.push_back(a.ref_id);
    // }
    // else if (a.type == Simplex::EDGE && b.type == Simplex::EDGE)
    // {
    //     finOverlaps.push_back(min(a.ref_id, b.ref_id));
    //     finOverlaps.push_back(max(a.ref_id, b.ref_id));
    // }
  }

  spdlog::trace("Total(filt.) overlaps: {:d}", finOverlaps.size() / 2);
  free(overlaps);
  // free(counter);
  // free(counter);
  cudaFree(d_count);
}
