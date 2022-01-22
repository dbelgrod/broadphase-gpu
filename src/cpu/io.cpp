#include <gpubf/io.hpp>

#include <igl/edges.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <vector>

#include <gpubf/aabb.hpp>

using namespace std;

namespace ccdcpu {

void constructBoxes(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
                    Eigen::MatrixXi &edges, Eigen::MatrixXi &faces,
                    vector<Aabb> &boxes) {
  tbb::task_scheduler_init init(64);
  addVertices(vertices_t0, vertices_t1, boxes);
  addEdges(vertices_t0, vertices_t1, edges, boxes);
  addFaces(vertices_t0, vertices_t1, faces, boxes);
  init.terminate();
}

void parseMesh(const char *filet0, const char *filet1, vector<Aabb> &boxes) {
  Eigen::MatrixXd V0;
  Eigen::MatrixXd V1;
  Eigen::MatrixXi F;

  string fn = string(filet0);
  string ext = fn.substr(fn.rfind('.') + 1);

  if (ext == "obj") {
    igl::readOBJ(filet0, V0, F);
    igl::readOBJ(filet1, V1, F);
  } else {
    igl::readPLY(filet0, V0, F);
    igl::readPLY(filet1, V1, F);
  }

  Eigen::MatrixXi E;
  igl::edges(F, E);
  // faces should be same F^{t=0} = F^{t=1}
  constructBoxes(V0, V1, E, F, boxes);
}

} // namespace ccdcpu