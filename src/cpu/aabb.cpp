#include <stq/cpu/aabb.hpp>

#include <tbb/parallel_for.h>

namespace stq::cpu {

void constructBoxes(const Eigen::MatrixXd &vertices_t0,
                    const Eigen::MatrixXd &vertices_t1,
                    const Eigen::MatrixXi &edges, const Eigen::MatrixXi &faces,
                    std::vector<Aabb> &boxes, double inflation_radius) {
  addVertices(vertices_t0, vertices_t1, boxes, inflation_radius);
  addEdges(boxes, edges, boxes);
  addFaces(boxes, faces, boxes);
}

float nextafter_up(float x) {
  return nextafterf(x, x + std::numeric_limits<float>::max());
};
float nextafter_down(float x) {
  return nextafterf(x, x - std::numeric_limits<float>::max());
};

void addVertices(const Eigen::MatrixXd &vertices_t0,
                 const Eigen::MatrixXd &vertices_t1, std::vector<Aabb> &boxes,
                 double inflation_radius) {
  size_t offset = boxes.size();
  boxes.resize(offset + vertices_t0.rows());

  tbb::parallel_for( //
    tbb::blocked_range<int>(0, vertices_t0.rows()),
    [&](const tbb::blocked_range<int> &r) {
      for (int i = r.begin(); i < r.end(); i++) {
        boxes[offset + i].id = offset + i;
        boxes[offset + i].vertexIds = {{i, -i - 1, -i - 1}};

        ArrayMax3 vertex_t0 = vertices_t0.row(i).cast<Scalar>();
        ArrayMax3 vertex_t1 = vertices_t1.row(i).cast<Scalar>();
#ifdef CCD_USE_DOUBLE
        boxes[offset + i].min = vertex_t0.min(vertex_t1) - inflation_radius;
        boxes[offset + i].max = vertex_t0.max(vertex_t1) + inflation_radius;
#else
        boxes[offset + i].min =
          vertex_t0.min(vertex_t1).unaryExpr(&nextafter_down) -
          nextafter_up(inflation_radius);
        boxes[offset + i].max =
          vertex_t0.max(vertex_t1).unaryExpr(&nextafter_up) +
          nextafter_up(inflation_radius);
#endif
      }
    });
}

void addEdges(const std::vector<Aabb> &vertex_boxes,
              const Eigen::MatrixXi &edges, std::vector<Aabb> &boxes) {
  size_t offset = boxes.size();
  boxes.resize(offset + edges.rows());

  tbb::parallel_for( //
    tbb::blocked_range<size_t>(0, edges.rows()),
    [&](const tbb::blocked_range<size_t> &r) {
      for (size_t i = r.begin(); i < r.end(); i++) {
        boxes[offset + i].id = offset + i;
        boxes[offset + i].vertexIds = {
          {edges(i, 0), edges(i, 1), -edges(i, 0) - 1}};

        const Aabb &v0_box = vertex_boxes[edges(i, 0)];
        const Aabb &v1_box = vertex_boxes[edges(i, 1)];

        boxes[offset + i].min = v0_box.min.min(v1_box.min);
        boxes[offset + i].max = v0_box.max.max(v1_box.max);
      }
    });
}

void addFaces(const std::vector<Aabb> &vertex_boxes,
              const Eigen::MatrixXi &faces, std::vector<Aabb> &boxes) {
  size_t offset = boxes.size();
  boxes.resize(offset + faces.rows());

  tbb::parallel_for( //
    tbb::blocked_range<size_t>(0, faces.rows()),
    [&](const tbb::blocked_range<size_t> &r) {
      for (size_t i = r.begin(); i < r.end(); i++) {
        boxes[offset + i].id = offset + i;
        boxes[offset + i].vertexIds = {{faces(i, 0), faces(i, 1), faces(i, 2)}};

        const Aabb &v0_box = vertex_boxes[faces(i, 0)];
        const Aabb &v1_box = vertex_boxes[faces(i, 1)];
        const Aabb &v2_box = vertex_boxes[faces(i, 2)];

        boxes[offset + i].min = v0_box.min.min(v1_box.min).min(v2_box.min);
        boxes[offset + i].max = v0_box.max.max(v1_box.max).max(v2_box.max);
      }
    });
}

} // namespace stq::cpu
