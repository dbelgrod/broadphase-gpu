#include <stq/cpu/io.hpp>

#include <vector>
#include <string>

#include <igl/edges.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>

#include <spdlog/spdlog.h>

#include <stq/cpu/aabb.hpp>

namespace stq::cpu {

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