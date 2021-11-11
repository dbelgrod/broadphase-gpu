#include <gpubf/aabb.cuh>


void addEdges
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    Eigen::MatrixXi& edges, 
    vector<Aabb>& boxes
)
{
    for (unsigned long i = 0; i < edges.rows(); i++) {
        Eigen::MatrixXf edge_vertex0_t0 = vertices_t0.cast<float>().row(edges(i, 0));
        Eigen::MatrixXf edge_vertex1_t0 = vertices_t0.cast<float>().row(edges(i, 1));
        Eigen::MatrixXf edge_vertex0_t1 = vertices_t1.cast<float>().row(edges(i, 0));
        Eigen::MatrixXf edge_vertex1_t1 = vertices_t1.cast<float>().row(edges(i, 1));

        Eigen::MatrixXf points(4, edge_vertex0_t0.size());
        points.row(0) = edge_vertex0_t0;
        points.row(1) = edge_vertex1_t0;
        points.row(2) = edge_vertex0_t1;
        points.row(3) = edge_vertex1_t1;

        int vertexIds[3] = {edges(i,0), edges(i,1), -edges(i,0)-1};

        Eigen::MatrixXf lower_bound = points.colwise().minCoeff();
        Eigen::MatrixXf upper_bound = points.colwise().maxCoeff();
        boxes.emplace_back(boxes.size(), i, Simplex::EDGE, vertexIds, lower_bound.array().data(), upper_bound.array().data());
    }
}

void addVertices
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    vector<Aabb>& boxes
)
{
    for (unsigned long i = 0; i < vertices_t0.rows(); i++) {
        Eigen::MatrixXf vertex_t0 = vertices_t0.cast<float>().row(i);
        Eigen::MatrixXf vertex_t1 = vertices_t1.cast<float>().row(i);

        Eigen::MatrixXf points(2, vertex_t0.size());
        points.row(0) = vertex_t0;
        points.row(1) = vertex_t1;

        int vertexIds[3] = {i, -i-1, -i-1};

        Eigen::MatrixXf lower_bound = points.colwise().minCoeff();
        Eigen::MatrixXf upper_bound = points.colwise().maxCoeff();
        boxes.emplace_back(boxes.size(), i, Simplex::VERTEX, vertexIds, lower_bound.array().data(), upper_bound.array().data());
    }
}

void addFaces
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    Eigen::MatrixXi& faces, 
    vector<Aabb>& boxes
)
{
    for (unsigned long i = 0; i < faces.rows(); i++) {
        Eigen::MatrixXf face_vertex0_t0 = vertices_t0.cast<float>().row(faces(i, 0));
        Eigen::MatrixXf face_vertex1_t0 = vertices_t0.cast<float>().row(faces(i, 1));
        Eigen::MatrixXf face_vertex2_t0 = vertices_t0.cast<float>().row(faces(i, 2)); 
        Eigen::MatrixXf face_vertex0_t1 = vertices_t1.cast<float>().row(faces(i, 0));
        Eigen::MatrixXf face_vertex1_t1 = vertices_t1.cast<float>().row(faces(i, 1)); 
        Eigen::MatrixXf face_vertex2_t1 = vertices_t1.cast<float>().row(faces(i, 2));

        Eigen::MatrixXf points(6, face_vertex0_t0.size());
        points.row(0) = face_vertex0_t0;
        points.row(1) = face_vertex1_t0;
        points.row(2) = face_vertex2_t0;
        points.row(3) = face_vertex0_t1;
        points.row(4) = face_vertex1_t1;
        points.row(5) = face_vertex2_t1;

        int vertexIds[3] = {faces(i,0), faces(i,1), faces(i,2)};

        Eigen::MatrixXf lower_bound = points.colwise().minCoeff();
        Eigen::MatrixXf upper_bound = points.colwise().maxCoeff();
        boxes.emplace_back(boxes.size(), i, Simplex::FACE, vertexIds, lower_bound.array().data(), upper_bound.array().data());
    }
};

