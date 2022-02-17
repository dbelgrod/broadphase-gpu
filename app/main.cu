#include <assert.h>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

// #define CCD_USE_DOUBLE

#include <gpubf/groundtruth.cuh>
#include <gpubf/simulation.cuh>
#include <gpubf/util.cuh>
// #include <gpubf/klee.cuh>
#include <gpubf/io.cuh>

#include <spdlog/spdlog.h>

using namespace std;
using namespace ccd::gpu;

// spdlog::set_level(spdlog::level::trace);

bool is_file_exist(const char *fileName) {
  ifstream infile(fileName);
  return infile.good();
}

int main(int argc, char **argv) {
  spdlog::set_level(static_cast<spdlog::level::level_enum>(0));
  vector<char *> compare;

  char *filet0;
  char *filet1;

  filet0 = argv[1];
  if (is_file_exist(argv[2]))
    filet1 = argv[2];
  else
    filet1 = argv[1];

  vector<Aabb> boxes;
  Eigen::MatrixXd vertices_t0;
  Eigen::MatrixXd vertices_t1;
  Eigen::MatrixXi faces;
  Eigen::MatrixXi edges;

  parseMesh(filet0, filet1, vertices_t0, vertices_t1, faces, edges);
  constructBoxes(vertices_t0, vertices_t1, edges, faces, boxes);
  size_t N = boxes.size();

  int nbox = 0;
  int parallel = 0;
  bool evenworkload = false;
  int devcount = 1;
  bool pairing = false;
  bool sharedqueue_mgpu = false;
  bool bigworkerqueue = false;

  int o;
  while ((o = getopt(argc, argv, "c:n:b:p:d:WPQZ")) != -1) {
    switch (o) {
    case 'c':
      optind--;
      for (; optind < argc && *argv[optind] != '-'; optind++) {
        compare.push_back(argv[optind]);
        // compare_mathematica(overlaps, argv[optind]);
      }
      break;
    case 'n':
      N = atoi(optarg);
      break;
    case 'b':
      nbox = atoi(optarg);
      break;
    case 'p':
      parallel = stoi(optarg);
      break;
    case 'd':
      devcount = atoi(optarg);
      break;
    case 'W':
      evenworkload = true;
      break;
    case 'P':
      pairing = true;
      break;
    case 'Q':
      sharedqueue_mgpu = true;
      break;
    case 'Z':
      bigworkerqueue = true;
      break;
    }
  }

  vector<pair<int, int>> overlaps;
  int2 *d_overlaps; // device
  int *d_count;     // device

  if (evenworkload)
    run_sweep_sharedqueue(boxes.data(), N, nbox, overlaps, d_overlaps, d_count,
                          parallel, devcount);
  else if (sharedqueue_mgpu)
    run_sweep_multigpu_queue(boxes.data(), N, nbox, overlaps, parallel,
                             devcount);
  else if (bigworkerqueue)
    run_sweep_bigworkerqueue(boxes.data(), N, nbox, overlaps, d_overlaps,
                             d_count, parallel, devcount);
  else
    run_sweep_multigpu(boxes.data(), N, nbox, overlaps, parallel, devcount);

  for (auto i : compare) {
    compare_mathematica(overlaps, i);
  }
}
