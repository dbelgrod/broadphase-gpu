#include <gpubf/aabb.cuh>
#include <gpubf/io.cuh>
#include <igl/edges.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <spdlog/spdlog.h>
#include <tbb/task_scheduler_init.h>

// using namespace ccdgpu;

void constructBoxes(const Eigen::MatrixXd &vertices_t0,
                    const Eigen::MatrixXd &vertices_t1,
                    const Eigen::MatrixXi &edges, const Eigen::MatrixXi &faces,
                    vector<ccdgpu::Aabb> &boxes,
                    ccdgpu::Scalar inflation_radius) {
  spdlog::trace("CPU_THREADS {}", CPU_THREADS);
  tbb::task_scheduler_init init(CPU_THREADS);
  addVertices(vertices_t0, vertices_t1, inflation_radius, boxes);
  addEdges(vertices_t0, vertices_t1, edges, inflation_radius, boxes);
  addFaces(vertices_t0, vertices_t1, faces, inflation_radius, boxes);
  init.terminate();
}

void parseMesh(const char *filet0, const char *filet1, Eigen::MatrixXd &V0,
               Eigen::MatrixXd &V1, Eigen::MatrixXi &F, Eigen::MatrixXi &E) {
  // Eigen::MatrixXd V0;
  // Eigen::MatrixXd V1;
  // Eigen::MatrixXi F;

  string fn = string(filet0);
  string ext = fn.substr(fn.rfind('.') + 1);

  if (ext == "obj") {
    igl::readOBJ(filet0, V0, F);
    igl::readOBJ(filet1, V1, F);
  } else {
    igl::readPLY(filet0, V0, F);
    igl::readPLY(filet1, V1, F);
  }

  // Eigen::MatrixXi E;
  igl::edges(F, E);
  // faces should be same F^{t=0} = F^{t=1}
  // constructBoxes(V0, V1, F, E, boxes);
}