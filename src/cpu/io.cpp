#include <stq/cpu/io.hpp>

#include <vector>
#include <string>

#include <igl/edges.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>

#include <tbb/global_control.h>

#include <spdlog/spdlog.h>

#include <stq/cpu/aabb.hpp>

namespace stq::cpu {

void constructBoxes(const Eigen::MatrixXd &vertices_t0,
                    const Eigen::MatrixXd &vertices_t1,
                    const Eigen::MatrixXi &edges, const Eigen::MatrixXi &faces,
                    std::vector<Aabb> &boxes, int threads) {
  if (threads <= 0)
    threads = CPU_THREADS;
  spdlog::trace("constructBoxes threads : {}", threads);
  tbb::global_control thread_limiter(
    tbb::global_control::max_allowed_parallelism, threads);
  addVertices(vertices_t0, vertices_t1, boxes);
  addEdges(vertices_t0, vertices_t1, edges, boxes);
  addFaces(vertices_t0, vertices_t1, faces, boxes);
}

void parseMesh(const char *filet0, const char *filet1,
               std::vector<Aabb> &boxes) {
  Eigen::MatrixXd V0;
  Eigen::MatrixXd V1;
  Eigen::MatrixXi F;

  std::string fn(filet0);
  std::string ext = fn.substr(fn.rfind('.') + 1);

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

} // namespace stq::cpu