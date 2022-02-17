#include <stq/gpu/io.cuh>

#include <string>

#include <igl/edges.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>

#include <spdlog/spdlog.h>

#include <tbb/global_control.h>

namespace stq::gpu {

void constructBoxes(const Eigen::MatrixXd &vertices_t0,
                    const Eigen::MatrixXd &vertices_t1,
                    const Eigen::MatrixXi &edges, const Eigen::MatrixXi &faces,
                    std::vector<Aabb> &boxes, int threads,
                    Scalar inflation_radius) {
  if (threads <= 0)
    threads = CPU_THREADS;
  spdlog::trace("constructBoxes threads : {}", threads);
  tbb::global_control thread_limiter(
    tbb::global_control::max_allowed_parallelism, threads);
  addVertices(vertices_t0, vertices_t1, inflation_radius, boxes);
  addEdges(vertices_t0, vertices_t1, edges, inflation_radius, boxes);
  addFaces(vertices_t0, vertices_t1, faces, inflation_radius, boxes);
}

void parseMesh(const char *filet0, const char *filet1, Eigen::MatrixXd &V0,
               Eigen::MatrixXd &V1, Eigen::MatrixXi &F, Eigen::MatrixXi &E) {
  // Eigen::MatrixXd V0;
  // Eigen::MatrixXd V1;
  // Eigen::MatrixXi F;

  std::string fn(filet0);
  std::string ext = fn.substr(fn.rfind('.') + 1);

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

} // namespace stq::gpu