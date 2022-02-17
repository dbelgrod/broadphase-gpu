#include <gpubf/aabb.hpp>

#include <tbb/parallel_for.h>

namespace ccd::cpu {

void merge_local_boxes(
  const tbb::enumerable_thread_specific<tbb::concurrent_vector<Aabb>> &storages,
  std::vector<Aabb> &boxes) {
  size_t num_boxes = boxes.size();
  for (const auto &local_boxes : storages) {
    num_boxes += local_boxes.size();
  }
  // serial merge!
  boxes.reserve(num_boxes);
  for (const auto &local_boxes : storages) {
    boxes.insert(boxes.end(), local_boxes.begin(), local_boxes.end());
  }
}

float nextafter_up(float x) { return nextafterf(x, x + 1.); };
float nextafter_down(float x) { return nextafterf(x, x - 1.); };

void addEdges(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
              Eigen::MatrixXi &edges, std::vector<Aabb> &boxes) {
  tbb::enumerable_thread_specific<tbb::concurrent_vector<Aabb>> storages;
  tbb::parallel_for(0, static_cast<int>(edges.rows()), 1, [&](int &i) {
    // for (unsigned long i = 0; i < edges.rows(); i++) {
    Eigen::MatrixXd edge_vertex0_t0 = vertices_t0.row(edges(i, 0));
    Eigen::MatrixXd edge_vertex1_t0 = vertices_t0.row(edges(i, 1));
    Eigen::MatrixXd edge_vertex0_t1 = vertices_t1.row(edges(i, 0));
    Eigen::MatrixXd edge_vertex1_t1 = vertices_t1.row(edges(i, 1));

    Eigen::MatrixXd points(4, edge_vertex0_t0.size());
    points.row(0) = edge_vertex0_t0;
    points.row(1) = edge_vertex1_t0;
    points.row(2) = edge_vertex0_t1;
    points.row(3) = edge_vertex1_t1;

    int vertexIds[3] = {edges(i, 0), edges(i, 1), -edges(i, 0) - 1};
#ifdef CCD_USE_DOUBLE
    Eigen::Vector3d lower_bound = points.colwise().minCoeff();
    Eigen::Vector3d upper_bound = points.colwise().maxCoeff();
#else
    Eigen::MatrixXf lower_bound =
        points.colwise().minCoeff().unaryExpr(&nextafter_down);
    Eigen::MatrixXf upper_bound =
        points.colwise().maxCoeff().unaryExpr(&nextafter_up);
#endif
    auto &local_boxes = storages.local();
    local_boxes.emplace_back(boxes.size() + i, vertexIds,
                             lower_bound.array().data(),
                             upper_bound.array().data());
  });
  merge_local_boxes(storages, boxes);
}

void addVertices(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
                 std::vector<Aabb> &boxes) {
  tbb::enumerable_thread_specific<tbb::concurrent_vector<Aabb>> storages;
  tbb::parallel_for(0, static_cast<int>(vertices_t0.rows()), 1, [&](int &i) {
    // for (unsigned long i = 0; i < vertices_t0.rows(); i++) {
    Eigen::MatrixXd vertex_t0 = vertices_t0.row(i);
    Eigen::MatrixXd vertex_t1 = vertices_t1.row(i);

    Eigen::MatrixXd points(2, vertex_t0.size());
    points.row(0) = vertex_t0;
    points.row(1) = vertex_t1;

    int vertexIds[3] = {i, -i - 1, -i - 1};

#ifdef CCD_USE_DOUBLE
    Eigen::MatrixXd lower_bound = points.colwise().minCoeff();
    Eigen::MatrixXd upper_bound = points.colwise().maxCoeff();
#else

    Eigen::MatrixXf lower_bound =
        points.colwise().minCoeff().unaryExpr(&nextafter_down);
    Eigen::MatrixXf upper_bound =
        points.colwise().maxCoeff().unaryExpr(&nextafter_up);
#endif
    auto &local_boxes = storages.local();
    local_boxes.emplace_back(boxes.size() + i, vertexIds,
                             lower_bound.array().data(),
                             upper_bound.array().data());
  });
  merge_local_boxes(storages, boxes);
}

void addFaces(Eigen::MatrixXd &vertices_t0, Eigen::MatrixXd &vertices_t1,
              Eigen::MatrixXi &faces, std::vector<Aabb> &boxes) {
  tbb::enumerable_thread_specific<tbb::concurrent_vector<Aabb>> storages;
  tbb::parallel_for(0, static_cast<int>(faces.rows()), 1, [&](int &i) {
    // for (unsigned long i = 0; i < faces.rows(); i++) {
    Eigen::MatrixXd face_vertex0_t0 = vertices_t0.row(faces(i, 0));
    Eigen::MatrixXd face_vertex1_t0 = vertices_t0.row(faces(i, 1));
    Eigen::MatrixXd face_vertex2_t0 = vertices_t0.row(faces(i, 2));
    Eigen::MatrixXd face_vertex0_t1 = vertices_t1.row(faces(i, 0));
    Eigen::MatrixXd face_vertex1_t1 = vertices_t1.row(faces(i, 1));
    Eigen::MatrixXd face_vertex2_t1 = vertices_t1.row(faces(i, 2));

    Eigen::MatrixXd points(6, face_vertex0_t0.size());
    points.row(0) = face_vertex0_t0;
    points.row(1) = face_vertex1_t0;
    points.row(2) = face_vertex2_t0;
    points.row(3) = face_vertex0_t1;
    points.row(4) = face_vertex1_t1;
    points.row(5) = face_vertex2_t1;

    int vertexIds[3] = {faces(i, 0), faces(i, 1), faces(i, 2)};

#ifdef CCD_USE_DOUBLE
    Eigen::Vector3d lower_bound = points.colwise().minCoeff();
    Eigen::Vector3d upper_bound = points.colwise().maxCoeff();
#else

        Eigen::MatrixXf lower_bound =
            points.colwise().minCoeff().unaryExpr(&nextafter_down);
        Eigen::MatrixXf upper_bound =
            points.colwise().maxCoeff().unaryExpr(&nextafter_up);
#endif
    auto &local_boxes = storages.local();
    local_boxes.emplace_back(boxes.size() + i, vertexIds,
                             lower_bound.array().data(),
                             upper_bound.array().data());
  });
  merge_local_boxes(storages, boxes);
}

} // namespace ccd::cpu
